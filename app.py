from pathlib import Path
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from train_model import run_experiment, load_data, save_pipeline, DATA_PATH

st.set_page_config(page_title="Ticket Classifier Lab", layout="wide")
st.title("AI Customer Support Ticket Classifier")
st.caption("Train locally, compare experiments, inspect errors, and add corrected examples.")

df = load_data()
st.sidebar.header("Experiment settings")
model_name = st.sidebar.selectbox(
    "Classification model",
    ["Logistic Regression", "Linear SVM", "Neural Network"]
)
vectorizer_name = st.sidebar.selectbox("Text features", ["TF-IDF", "Count"])
ngram_label = st.sidebar.selectbox("N-grams", ["Unigrams (1)", "Unigrams + bigrams (1–2)"])
ngram_max = 1 if ngram_label == "Unigrams (1)" else 2
remove_stop_words = st.sidebar.checkbox("Remove English stop words", value=False)
strength = st.sidebar.select_slider(
    "Regularization strength (C; ignored by neural network)",
    options=[0.1, 0.5, 1.0, 2.0, 5.0], value=1.0
)

st.subheader("Dataset")
a, b, c = st.columns(3)
a.metric("Labeled tickets", len(df))
b.metric("Categories", df["category"].nunique())
c.metric("Examples per category (minimum)", int(df["category"].value_counts().min()))
with st.expander("View current dataset"):
    st.dataframe(df, use_container_width=True)

if "history" not in st.session_state:
    st.session_state.history = []
if "latest" not in st.session_state:
    st.session_state.latest = None

if st.button("Train / run experiment", type="primary"):
    with st.spinner("Training and evaluating model..."):
        result = run_experiment(
            model_name, vectorizer_name, ngram_max,
            remove_stop_words, strength
        )
        st.session_state.latest = result
        st.session_state.history.append({
            "experiment": len(st.session_state.history) + 1,
            "model": model_name,
            "features": vectorizer_name,
            "ngram_max": ngram_max,
            "stop_words_removed": remove_stop_words,
            "strength_C": strength,
            "accuracy": result["accuracy"],
            "macro_f1": result["report"]["macro avg"]["f1-score"]
        })
        save_pipeline(result["pipeline"])

result = st.session_state.latest
if result is not None:
    st.subheader("Latest evaluation")
    x, y, z = st.columns(3)
    x.metric("Accuracy", f'{result["accuracy"]:.1%}')
    y.metric("Macro F1-score", f'{result["report"]["macro avg"]["f1-score"]:.3f}')
    z.metric("Test tickets", result["test_size"])

    report_df = pd.DataFrame(result["report"]).T
    st.markdown("#### Per-category precision, recall and F1")
    st.dataframe(report_df.round(3), use_container_width=True)

    st.markdown("#### Confusion matrix")
    fig, ax = plt.subplots()
    matrix = result["confusion_matrix"]
    image = ax.imshow(matrix)
    ax.set_xticks(range(len(result["labels"])), result["labels"], rotation=35, ha="right")
    ax.set_yticks(range(len(result["labels"])), result["labels"])
    ax.set_xlabel("Predicted category")
    ax.set_ylabel("Actual category")
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            ax.text(j, i, str(value), ha="center", va="center")
    fig.colorbar(image, ax=ax)
    st.pyplot(fig)
    plt.close(fig)

    errors = pd.DataFrame({
        "ticket": result["test_texts"],
        "actual": result["actual"],
        "predicted": result["predicted"]
    })
    errors = errors[errors["actual"] != errors["predicted"]]
    st.markdown("#### Misclassified tickets")
    if errors.empty:
        st.success("No errors on this test split. Try a harder or larger dataset.")
    else:
        st.dataframe(errors, use_container_width=True)

st.subheader("Experiment history")
if st.session_state.history:
    history = pd.DataFrame(st.session_state.history)
    st.dataframe(history, use_container_width=True)
    chart_df = history.set_index("experiment")[["accuracy", "macro_f1"]]
    st.line_chart(chart_df)
else:
    st.info("Run at least one experiment to begin comparing results.")

st.divider()
st.subheader("Predict a new ticket")
ticket = st.text_area("Enter a customer ticket", placeholder="Example: I paid twice for my order")
if st.button("Predict category"):
    if not ticket.strip():
        st.warning("Enter some ticket text first.")
    elif result is None:
        st.warning("Run an experiment first so a model is trained.")
    else:
        prediction = result["pipeline"].predict([ticket.strip()])[0]
        st.success(f"Predicted category: {prediction}")

st.divider()
st.subheader("Add a corrected labeled example")
st.write("Use this when a ticket was incorrectly classified. Choose the correct label before saving.")
new_text = st.text_area("Ticket text to add", key="new_text")
categories = sorted(df["category"].unique())
new_category = st.selectbox("Correct category", categories)
if st.button("Save example to dataset"):
    if not new_text.strip():
        st.warning("Enter ticket text.")
    elif new_text.strip().casefold() in set(df["text"].str.casefold()):
        st.warning("This exact ticket already exists in the dataset.")
    else:
        new_row = pd.DataFrame([{"text": new_text.strip(), "category": new_category}])
        new_row.to_csv(DATA_PATH, mode="a", header=False, index=False)
        st.success("Saved. Run a new experiment to retrain and compare.")
        st.rerun()
