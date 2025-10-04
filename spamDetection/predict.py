from joblib import load

def predict_from_file():
    pipeline = load('spam.joblib')
    with open('test.txt', 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]
    preds = pipeline.predict(lines)
    probs = pipeline.predict_proba(lines) if hasattr(pipeline, "predict_proba") else None
    for i, text in enumerate(lines):
        label = 'spam' if preds[i]==1 else 'ham'
        prob = probs[i][1] if probs is not None else None
        print(f"Text: {text[:120]!r}\nPredicted: {label} (spam_prob={prob})\n")

predict_from_file()
