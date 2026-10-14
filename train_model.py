from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "tickets.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

def load_data():
    df = pd.read_csv(DATA_PATH).dropna(subset=["text", "category"])
    df["text"] = df["text"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip()
    df = df[(df["text"] != "") & (df["category"] != "")]
    return df

def make_pipeline(model_name="Logistic Regression",
                  vectorizer_name="TF-IDF",
                  ngram_max=2,
                  remove_stop_words=False,
                  strength=1.0,
                  max_iter=500):
    stop_words = "english" if remove_stop_words else None

    if vectorizer_name == "Count":
        vectorizer = CountVectorizer(
            ngram_range=(1, ngram_max), stop_words=stop_words,
            min_df=1
        )
    else:
        vectorizer = TfidfVectorizer(
            ngram_range=(1, ngram_max), stop_words=stop_words,
            min_df=1, sublinear_tf=True
        )

    if model_name == "Logistic Regression":
        classifier = LogisticRegression(
            C=float(strength), max_iter=int(max_iter),
            class_weight=None, random_state=42
        )
    elif model_name == "Linear SVM":
        classifier = LinearSVC(
            C=float(strength), class_weight=None,
            random_state=42, max_iter=5000
        )
    else:
        classifier = MLPClassifier(
            hidden_layer_sizes=(50,),
            alpha=0.0001,
            max_iter=1000,
            early_stopping=False,
            random_state=42,
            learning_rate_init=0.001,
            learning_rate='adaptive',
            batch_size='auto',
            n_iter_no_change=10
        )

    if model_name == "Neural Network":
        return Pipeline([
            ("vectorizer", vectorizer),
            ("scaler", StandardScaler(with_mean=False)),
            ("classifier", classifier)
        ])
    else:
        return Pipeline([("vectorizer", vectorizer), ("classifier", classifier)])

def run_experiment(model_name="Logistic Regression",
                   vectorizer_name="TF-IDF",
                   ngram_max=2,
                   remove_stop_words=False,
                   strength=1.0):
    df = load_data()
    counts = df["category"].value_counts()
    if len(counts) < 2 or counts.min() < 2:
        raise ValueError("Each category needs at least two labeled examples.")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["category"], test_size=0.25,
        random_state=42, stratify=df["category"]
    )

    pipeline = make_pipeline(
        model_name, vectorizer_name, ngram_max,
        remove_stop_words, strength
    )
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    labels = sorted(df["category"].unique())
    report = classification_report(
        y_test, predictions, labels=labels,
        output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(y_test, predictions, labels=labels)

    results = {
        "accuracy": accuracy_score(y_test, predictions),
        "report": report,
        "labels": labels,
        "confusion_matrix": matrix.tolist(),
        "test_texts": X_test.tolist(),
        "actual": y_test.tolist(),
        "predicted": predictions.tolist(),
        "model_name": model_name,
        "vectorizer_name": vectorizer_name,
        "ngram_max": ngram_max,
        "remove_stop_words": remove_stop_words,
        "strength": strength,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "pipeline": pipeline,
    }
    return results

def save_pipeline(pipeline):
    joblib.dump(pipeline, MODEL_DIR / "ticket_classifier.joblib")

if __name__ == "__main__":
    result = run_experiment()
    save_pipeline(result["pipeline"])
    print(f"Accuracy: {result['accuracy']:.3f}")
    print(pd.DataFrame(result["report"]).T.round(3))
    print("Saved model to models/ticket_classifier.joblib")
