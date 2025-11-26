import io

import pandas as pd
import streamlit as st

from nlp_utils import (
    basic_clean,
    split_documents,
    build_tfidf,
    cluster_documents,
    extract_top_keywords_per_cluster,
    compute_sentiment_scores,
)

st.set_page_config(
    page_title="TextInsight – NLP Text Explorer",
    layout="wide"
)


def main():
    st.title("🧠 TextInsight – NLP Text Explorer")
    st.write(
        "Upload a text file, explore **topics**, **keywords**, and **sentiment** "
        "using classic NLP (TF-IDF + K-Means + VADER sentiment)."
    )

    with st.sidebar:
        st.header("⚙️ Settings")
        n_clusters = st.slider(
            "Number of clusters (topics)",
            min_value=2,
            max_value=10,
            value=4,
            step=1,
            help="How many groups of similar documents you want."
        )

        show_raw = st.checkbox("Show raw documents", value=False)
        show_clean = st.checkbox("Show cleaned text", value=False)

    uploaded_file = st.file_uploader(
        "Upload a .txt file with multiple documents",
        type=["txt"],
        help=(
            "You can separate documents by blank lines, or put one document per line."
        )
    )

    if not uploaded_file:
        st.info("⬆️ Upload a text file to begin.")
        st.stop()

    # Read the file content
    bytes_data = uploaded_file.read()
    try:
        text_data = bytes_data.decode("utf-8")
    except UnicodeDecodeError:
        text_data = bytes_data.decode("latin-1")

    docs_raw = split_documents(text_data)

    if len(docs_raw) < 2:
        st.error("I only found one document. Add multiple documents (lines or blank lines).")
        st.stop()

    st.success(f"Loaded **{len(docs_raw)}** documents from the file ✅")

    # Cleaned docs
    docs_clean = [basic_clean(d) for d in docs_raw]

    if show_raw:
        with st.expander("📄 Raw Documents"):
            for i, doc in enumerate(docs_raw):
                st.markdown(f"**Doc {i}**")
                st.text(doc)
                st.markdown("---")

    if show_clean:
        with st.expander("🧼 Cleaned Documents"):
            for i, doc in enumerate(docs_clean):
                st.markdown(f"**Doc {i}**")
                st.text(doc)
                st.markdown("---")

    # Build TF-IDF and cluster
    vectorizer, tfidf_matrix = build_tfidf(docs_clean)
    model = cluster_documents(tfidf_matrix, n_clusters=n_clusters)
    labels = model.labels_
    cluster_keywords = extract_top_keywords_per_cluster(model, vectorizer, top_n=8)
    sentiments = compute_sentiment_scores(docs_raw)

    # Build DataFrame
    df = pd.DataFrame({
        "document_id": list(range(len(docs_raw))),
        "cluster": labels,
        "sentiment": sentiments,
        "raw_text": docs_raw,
        "clean_text": docs_clean,
    })

    st.subheader("📊 Cluster Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Documents per cluster**")
        cluster_counts = df["cluster"].value_counts().sort_index()
        st.bar_chart(cluster_counts)

    with col2:
        st.write("**Average sentiment per cluster**")
        avg_sentiment = df.groupby("cluster")["sentiment"].mean().sort_index()
        st.bar_chart(avg_sentiment)

    st.subheader("🗂 Topic Keywords")
    for cluster_id in sorted(cluster_keywords.keys()):
        with st.expander(f"Cluster {cluster_id} – top keywords"):
            st.write(", ".join(cluster_keywords[cluster_id]))

    st.subheader("📑 Document Table")
    st.write(
        "You can sort/filter documents, inspect their raw text, cluster and sentiment."
    )

    display_df = df[["document_id", "cluster", "sentiment", "raw_text"]]
    st.dataframe(display_df)

    # Option to download results
    st.subheader("⬇️ Download Results")
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="Download CSV",
        data=csv_buffer.getvalue(),
        file_name="textinsight_results.csv",
        mime="text/csv"
    )

    st.caption(
        "Built with ❤️ using Streamlit, scikit-learn, and NLTK VADER. "
        "Perfect for quick text exploration, topic discovery, and basic sentiment analysis."
    )


if __name__ == "__main__":
    main()
