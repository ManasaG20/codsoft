import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from joblib import dump
from sklearn.utils import shuffle

def build_pipeline():
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(
            lowercase=True,
            strip_accents='unicode',
            analyzer='word',
            ngram_range=(1,2),
            min_df=2
        )),
        ('clf', LogisticRegression(
            solver='liblinear',
            class_weight='balanced', 
            max_iter=1000,
            random_state=42
        ))
    ])
    return pipe

def main():
    df = pd.read_csv('spam.csv', encoding='latin-1')
    df = df[['v1', 'v2']].rename(columns={'v1':'label', 'v2':'message'})
    df = df.dropna()
    df['label'] = df['label'].str.strip().str.lower()
    df = shuffle(df, random_state=42).reset_index(drop=True)
    print(f"Loaded {len(df)} rows. Label distribution:\n{df['label'].value_counts()}\n")

    X = df['message'].astype(str)
    y = df['label'].map(lambda x: 1 if x in ['spam', 'spam\r', 'spam\n'] else 0) 

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipeline = build_pipeline()

   
    param_grid = {
        'clf__C': [0.01, 0.1, 1.0, 5.0]
    }
    print("Starting GridSearchCV (small grid)...")
    gs = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring='f1', verbose=1)
    gs.fit(X_train, y_train)
    print(f"Best params: {gs.best_params_}")
    best_pipeline = gs.best_estimator_

    # Evaluate
    y_pred = best_pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc:.4f}\n")
    print("Classification report:\n")
    print(classification_report(y_test, y_pred, target_names=['ham','spam']))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    dump(best_pipeline,'spam.joblib')
    print(f"\nSaved pipeline to: {'spam.joblib'}")

main()
