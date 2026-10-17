from pathlib import Path
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from train_model import run_experiment, load_data, save_pipeline, DATA_PATH

# Optional torch import for transformer support
TRANSFORMER_AVAILABLE = False
try:
    import torch
    from finetune_model import load_finetuned_model, predict_category, finetune_model
    TRANSFORMER_AVAILABLE = True
except (ImportError, OSError) as e:
    st.warning(f"Transformer support unavailable: {str(e)}")

st.set_page_config(page_title="Ticket Classifier Lab", layout="wide")
st.title("AI Customer Support Ticket Classifier")
st.caption("Train locally, compare experiments, inspect errors, and add corrected examples.")

df = load_data()
st.sidebar.header("Experiment settings")

model_options = ["Traditional ML"]
if TRANSFORMER_AVAILABLE:
    model_options.append("Transformer (Fine-tuned)")

model_type = st.sidebar.radio(
    "Model Type",
    model_options,
    help="Traditional ML uses scikit-learn models. Transformer uses fine-tuned DistilBERT.",
    key="model_type_radio"
)

if model_type == "Traditional ML":
    model_name = st.sidebar.selectbox(
        "Classification model",
        ["Logistic Regression", "Linear SVM", "Neural Network"],
        key="model_select"
    )
    vectorizer_name = st.sidebar.selectbox("Text features", ["TF-IDF", "Count"], key="vectorizer_select")
    ngram_label = st.sidebar.selectbox("N-grams", ["Unigrams (1)", "Unigrams + bigrams (1–2)"], key="ngram_select")
    ngram_max = 1 if ngram_label == "Unigrams (1)" else 2
    remove_stop_words = st.sidebar.checkbox("Remove English stop words", value=False, key="stopwords_checkbox")
    strength = st.sidebar.select_slider(
        "Regularization strength (C; ignored by neural network)",
        options=[0.1, 0.5, 1.0, 2.0, 5.0], value=1.0,
        key="strength_slider"
    )
else:
    model_name = "DistilBERT"
    vectorizer_name = "Transformer"
    ngram_max = 1
    remove_stop_words = False
    strength = 1.0

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
    if model_type == "Traditional ML":
        with st.spinner("Training and evaluating model..."):
            result = run_experiment(
                model_name, vectorizer_name, ngram_max,
                remove_stop_words, strength
            )
            st.session_state.latest = result
            st.session_state.history.append({
                "experiment": len(st.session_state.history) + 1,
                "model_type": "Traditional ML",
                "model": model_name,
                "features": vectorizer_name,
                "ngram_max": ngram_max,
                "stop_words_removed": remove_stop_words,
                "strength_C": strength,
                "accuracy": result["accuracy"],
                "macro_f1": result["report"]["macro avg"]["f1-score"]
            })
            save_pipeline(result["pipeline"])
    elif TRANSFORMER_AVAILABLE:
        with st.spinner("Fine-tuning transformer model (this may take several minutes)..."):
            try:
                result = finetune_model()
                st.session_state.latest = {
                    "accuracy": result["accuracy"],
                    "macro_f1": result["macro_f1"],
                    "model_name": model_name,
                    "vectorizer_name": vectorizer_name,
                    "ngram_max": ngram_max,
                    "remove_stop_words": remove_stop_words,
                    "strength": strength,
                    "model_path": result["model_path"],
                    "label2id": result["label2id"],
                    "id2label": result["id2label"]
                }
                st.session_state.history.append({
                    "experiment": len(st.session_state.history) + 1,
                    "model_type": "Transformer",
                    "model": model_name,
                    "features": vectorizer_name,
                    "ngram_max": ngram_max,
                    "stop_words_removed": remove_stop_words,
                    "strength_C": strength,
                    "accuracy": result["accuracy"],
                    "macro_f1": result["macro_f1"]
                })
                st.success("Fine-tuning complete!")
            except Exception as e:
                st.error(f"Fine-tuning failed: {str(e)}")

result = st.session_state.latest
if result is not None:
    st.subheader("Latest evaluation")
    x, y = st.columns(2)
    x.metric("Accuracy", f'{result["accuracy"]:.1%}')

    # Handle different result structures for ML vs Transformer
    if "macro_f1" in result:
        macro_f1 = result["macro_f1"]
    elif "report" in result and "macro avg" in result["report"]:
        macro_f1 = result["report"]["macro avg"]["f1-score"]
    else:
        macro_f1 = 0.0

    y.metric("Macro F1-score", f'{macro_f1:.3f}')

    if model_type == "Traditional ML":
        z = st.columns(1)[0]
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
    else:
        st.info("Transformer model evaluation metrics shown above. Detailed per-category metrics and confusion matrix are not available for transformer fine-tuning in this version.")

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
        if model_type == "Traditional ML":
            prediction = result["pipeline"].predict([ticket.strip()])[0]
            st.success(f"Predicted category: {prediction}")
        elif TRANSFORMER_AVAILABLE:
            try:
                model, tokenizer = load_finetuned_model()
                prediction, confidence = predict_category(ticket.strip(), model, tokenizer)
                st.success(f"Predicted category: {prediction}")
                st.info(f"Confidence: {confidence:.2%}")
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

st.divider()
st.subheader("Add a corrected labeled example")
st.write("Use this when a ticket was incorrectly classified. Choose the correct label before saving.")
new_text = st.text_area("Ticket text to add", key="new_text")
categories = sorted(df["category"].unique())
new_category = st.selectbox("Correct category", categories, key="category_select")
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
