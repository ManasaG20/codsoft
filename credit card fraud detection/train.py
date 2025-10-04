import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from joblib import dump

def load_data(file_path='dataset.csv'):
    df = pd.read_csv(file_path)
    return df

def build_pipeline(numeric_features, categorical_features):
    # Preprocessing for numeric features
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # Preprocessing for categorical features
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Complete pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', LogisticRegression(
            solver='liblinear',
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        ))
    ])
    return pipeline

def main():
    df = load_data() 
    print(f"Loaded {len(df)} rows")

    # Drop columns that are not useful
    X = df.drop(columns=['Exited', 'RowNumber', 'CustomerId', 'Surname'])
    y = df['Exited']

    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    pipeline = build_pipeline(numeric_features, categorical_features)

    # Split into train-test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training model...")
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Save model
    dump(pipeline, "customer_churn_model.joblib")

if __name__ == "__main__":
    main()
