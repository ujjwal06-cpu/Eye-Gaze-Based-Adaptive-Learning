"""
Week 3 -- Feature Extraction
==============================
Third stage: Tokenization -> [Feature Extraction] -> Intent Classification

Turns cleaned text into numeric vectors a classifier can actually use.
We use TF-IDF (Term Frequency - Inverse Document Frequency): each
word gets weighted by how often it appears in a given sentence,
discounted by how common that word is across ALL sentences.

Why TF-IDF over plain word counts: words like "please" or "i" show up
in almost every intent, so they shouldn't count for much. Words like
"thirsty" or "restroom" are rarer and much more informative -- TF-IDF
automatically gives those more weight without us hand-picking them.

sklearn's TfidfVectorizer already does cleaning + tokenizing + TF-IDF
weighting internally, but we plug in OUR OWN clean_and_tokenize() from
text_cleaning.py / tokenization.py so all three stages stay consistent
and visible as separate, explainable steps (which matters for
presenting the pipeline, not just running it).
"""

from sklearn.feature_extraction.text import TfidfVectorizer

from tokenization import clean_and_tokenize


def build_vectorizer() -> TfidfVectorizer:
    """Create a TF-IDF vectorizer wired to our own cleaning/tokenizing."""
    return TfidfVectorizer(
        tokenizer=clean_and_tokenize,
        preprocessor=lambda x: x,  # skip sklearn's default preprocessing;
        token_pattern=None,        # we already handled cleaning ourselves
    )


def extract_features(texts: list[str]):
    """Fit a TF-IDF vectorizer on `texts` and return (vectorizer, matrix)."""
    vectorizer = build_vectorizer()
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


if __name__ == "__main__":
    samples = [
        "I need water",
        "I am thirsty",
        "I need food",
        "I am hungry",
    ]
    vectorizer, matrix = extract_features(samples)
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"Feature matrix shape: {matrix.shape}  (n_samples x n_features)")
    print("\nVocabulary:", sorted(vectorizer.vocabulary_.keys()))
    print("\nFeature vector for 'I need water':")
    row = matrix[0].toarray()[0]
    for word, idx in sorted(vectorizer.vocabulary_.items(), key=lambda kv: kv[1]):
        if row[idx] > 0:
            print(f"  {word:10} {row[idx]:.3f}")
