import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from joblib import dump

def load_data(file_path):
    """
    Load data from train.txt or test.txt
    Format: id ::: title ::: genre ::: plot
    """
    rows = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(":::")
            if len(parts) >= 4:
                idx = parts[0].strip()
                title = parts[1].strip()
                genre = parts[2].strip().lower()
                plot = ":::".join(parts[3:]).strip()
                rows.append((idx, title, genre, plot))
    df = pd.DataFrame(rows, columns=["id", "title", "genre", "plot"])
    return df

def build_pipeline():
    """
    Build TF-IDF + Logistic Regression (multi-class via OneVsRest)
    """
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            lowercase=True,
            strip_accents='unicode',
            analyzer='word',
            ngram_range=(1,2),
            min_df=2
        )),
        ('clf', OneVsRestClassifier(
            LogisticRegression(
                solver='liblinear',
                class_weight='balanced',
                max_iter=2000,
                random_state=42
            )
        ))
    ])
    return pipeline

def main():
    # Load train and test
    train_df = load_data("train_data.txt")
    test_df = load_data("test_data.txt")

    print(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows")
    print(f"Number of genres: {train_df['genre'].nunique()}")

    # Optionally filter very rare genres in training
    genre_counts = train_df['genre'].value_counts()
    train_df = train_df[train_df['genre'].isin(genre_counts[genre_counts>=5].index)].reset_index(drop=True)
    test_df = test_df[test_df['genre'].isin(train_df['genre'].unique())].reset_index(drop=True)

    X_train = train_df['plot'].astype(str)
    y_train = train_df['genre']
    X_test = test_df['plot'].astype(str)
    y_test = test_df['genre']

    pipeline = build_pipeline()
    print("Training model...")
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc:.4f}")
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Save model
    dump(pipeline, "movie.joblib")
    print("\nSaved pipeline to movie_genre_model.joblib")

if __name__ == "__main__":
    main()
