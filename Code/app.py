# ---------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------
import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import plotly.graph_objects as go
import seaborn as sns
from collections import Counter
import itertools

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# ---------------------------------------------------------
# LOAD BACKEND OBJECTS
# ---------------------------------------------------------
df = joblib.load("df.pkl")
dictionary = joblib.load("dictionary.pkl")
lda_model = joblib.load("lda_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
tfidf_matrix = joblib.load("tfidf_matrix.pkl")

# ---------------------------------------------------------
# TEXT PREPROCESSING
# ---------------------------------------------------------
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if pd.isna(text):
        return []
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    tokens = text.split()
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and len(word) > 2
    ]
    return tokens

# ---------------------------------------------------------
# SEARCH FUNCTIONS
# ---------------------------------------------------------
def get_query_topic(query):
    query_clean = clean_text(query)
    bow = dictionary.doc2bow(query_clean)
    return lda_model.get_document_topics(bow)

def get_dominant_topic(topic_dist):
    if not topic_dist:
        return None
    return max(topic_dist, key=lambda x: x[1])[0]

def topic_routed_search(query, top_n=15):
    topic_dist = get_query_topic(query)
    dom_topic = get_dominant_topic(topic_dist)

    if dom_topic is None:
        return df.head(0)

    topic_docs = df[df["dominant_topic"] == dom_topic].copy()

    query_clean = " ".join(clean_text(query))
    q_vec = vectorizer.transform([query_clean])
    sims = cosine_similarity(q_vec, tfidf_matrix[topic_docs.index]).flatten()

    topic_docs["similarity"] = sims
    return topic_docs.sort_values("similarity", ascending=False).head(top_n)

# ---------------------------------------------------------
# SANKEY DIAGRAM (Keyword → Topic)
# ---------------------------------------------------------
def build_sankey(data, max_keywords=20):
    if data.empty or "keywords" not in data.columns or "topic_name" not in data.columns:
        return go.Figure()

    df_local = data.copy()
    df_local["keywords_list"] = df_local["keywords"].apply(
        lambda x: x if isinstance(x, list) else str(x).split(",")
    )

    exploded = df_local.explode("keywords_list")
    exploded["keywords_list"] = exploded["keywords_list"].astype(str).str.strip().str.lower()

    top_keywords = (
        exploded["keywords_list"]
        .value_counts()
        .head(max_keywords)
        .index
        .tolist()
    )

    if not top_keywords:
        return go.Figure()

    filtered = exploded[exploded["keywords_list"].isin(top_keywords)]

    keyword_nodes = top_keywords
    topic_nodes = df_local["topic_name"].dropna().unique().tolist()
    all_nodes = keyword_nodes + topic_nodes

    node_index = {name: i for i, name in enumerate(all_nodes)}

    links = (
        filtered.groupby(["keywords_list", "topic_name"])
        .size()
        .reset_index(name="count")
    )

    source = links["keywords_list"].map(node_index)
    target = links["topic_name"].map(node_index)
    value = links["count"]

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=20,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=all_nodes,
                    color="rgba(31, 119, 180, 0.8)",
                ),
                link=dict(
                    source=source,
                    target=target,
                    value=value,
                    color="rgba(31, 119, 180, 0.4)",
                ),
            )
        ]
    )

    fig.update_layout(
        title_text="Keyword → Topic Flow",
        font_size=12,
        height=700,
        width=1200
    )

    return fig

# ---------------------------------------------------------
# THREE-COLUMN ALLUVIAL DIAGRAM (Term1 → Term3 → Term2)
# ---------------------------------------------------------
def build_three_column_alluvial(df, kw_col="author_kw_list"):
    if df.empty or kw_col not in df.columns:
        return go.Figure()

    kw_lists = df[kw_col].dropna().tolist()
    if not kw_lists:
        return go.Figure()

    freq_counter = Counter()
    for kws in kw_lists:
        freq_counter.update(kws)

    freq_df = pd.DataFrame(freq_counter.items(), columns=["kw", "freq"])
    freq_df = freq_df.sort_values("freq", ascending=False).reset_index(drop=True)

    top45 = freq_df.head(45)["kw"].tolist()
    if len(top45) < 45:
        return go.Figure()

    Term1 = top45[0:15]
    Term3 = top45[15:30]
    Term2 = top45[30:45]

    pair_counter = Counter()
    for kws in kw_lists:
        kws_unique = sorted(set(kws))
        for a, b in itertools.combinations(kws_unique, 2):
            pair_counter[(a, b)] += 1

    pairs_df = pd.DataFrame(
        [(a, b, w) for (a, b), w in pair_counter.items()],
        columns=["term1", "term2", "weight"]
    )

    flows_13 = pairs_df[
        pairs_df["term1"].isin(Term1) & pairs_df["term2"].isin(Term3)
    ]
    flows_32 = pairs_df[
        pairs_df["term1"].isin(Term3) & pairs_df["term2"].isin(Term2)
    ]

    all_nodes = Term1 + Term3 + Term2
    node_to_id = {n: i for i, n in enumerate(all_nodes)}

    sources, targets, values, hover_texts = [], [], [], []
    freq_map = dict(zip(freq_df.kw, freq_df.freq))

    for _, r in flows_13.iterrows():
        a, b, w = r["term1"], r["term2"], r["weight"]
        sources.append(node_to_id[a])
        targets.append(node_to_id[b])
        values.append(w)
        hover_texts.append(
            f"<b>{a}</b> → <b>{b}</b><br>"
            f"Co-occurrence: {w}<br>"
            f"{a} freq: {freq_map.get(a, 0)}<br>"
            f"{b} freq: {freq_map.get(b, 0)}"
        )

    for _, r in flows_32.iterrows():
        a, b, w = r["term1"], r["term2"], r["weight"]
        sources.append(node_to_id[a])
        targets.append(node_to_id[b])
        values.append(w)
        hover_texts.append(
            f"<b>{a}</b> → <b>{b}</b><br>"
            f"Co-occurrence: {w}<br>"
            f"{a} freq: {freq_map.get(a, 0)}<br>"
            f"{b} freq: {freq_map.get(b, 0)}"
        )

    palette = sns.color_palette("tab20", len(all_nodes))
    node_colors = [
        f"rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, 0.85)"
        for r, g, b in palette
    ]

    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            pad=20,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=all_nodes,
            color=node_colors,
            hovertemplate="<b>%{label}</b><extra></extra>"
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color="rgba(150,150,150,0.25)",
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hover_texts
        )
    )])

    fig.update_layout(
        title_text="Three‑Column Alluvial Diagram (Top 45 Keywords)",
        font_size=12,
        width=1500,
        height=900
    )

    return fig

# ---------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------
st.set_page_config(page_title="Research Paper Search", layout="wide")

st.title("🔎 Topic‑Routed Research Paper Search")
st.write("Search your corpus using LDA topic routing + TF‑IDF ranking.")

query = st.text_input("Enter your query", placeholder="e.g., digital health apps for chronic disease")
top_n = st.slider("Number of results", 5, 50, 15)

if "last_results" not in st.session_state:
    st.session_state["last_results"] = df

if st.button("Search"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Searching..."):
            results = topic_routed_search(query, top_n=top_n)

        if results.empty:
            st.error("No results found.")
        else:
            st.session_state["last_results"] = results

            display_cols = [
                "title", "authors", "pub_year", "journal",
                "similarity", "dominant_topic", "topic_name",
                "topic_top_words", "keywords"
            ]
            valid_cols = [c for c in display_cols if c in results.columns]

            st.subheader("Search Results")
            st.dataframe(results[valid_cols], use_container_width=True)

            def to_excel(df_):
                output = BytesIO()
                writer = pd.ExcelWriter(output, engine='xlsxwriter')
                df_.to_excel(writer, index=False, sheet_name='Results')
                writer.close()
                return output.getvalue()

            excel_data = to_excel(results[valid_cols])

            st.download_button(
                label="📥 Download Results as Excel",
                data=excel_data,
                file_name="search_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

# ---------------------------------------------------------
# VISUALIZATIONS
# ---------------------------------------------------------
st.subheader("Keyword → Topic Sankey Diagram")
fig_sankey = build_sankey(st.session_state["last_results"])
st.plotly_chart(fig_sankey, use_container_width=True)

st.subheader("Three‑Column Alluvial Diagram (Top 45 Keywords)")
fig_alluvial = build_three_column_alluvial(st.session_state["last_results"])
st.plotly_chart(fig_alluvial, use_container_width=True)