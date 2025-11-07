from typing import List, Dict, Any
import spacy as spacy_mod
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer

class ClassicalAgent:
    def __init__(self):
        # spaCy model
        try:
            self.nlp = spacy_mod.load("en_core_web_sm")
        except OSError:
            import spacy.cli as spacy_cli
            spacy_cli.download("en_core_web_sm")
            self.nlp = spacy_mod.load("en_core_web_sm")

        # VADER sentiment
        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            nltk.download("vader_lexicon")
        self.sia = SentimentIntensityAnalyzer()

        # TF-IDF
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1,2))
        self.feature_names: List[str] = []

    def fit_tfidf_vectorizer(self, corpus: List[str]):
        self.vectorizer.fit(corpus)
        self.feature_names = self.vectorizer.get_feature_names_out().tolist()

    def analyze_documents(self, docs: List[str], top_k_keywords: int = 5) -> List[Dict[str, Any]]:
        spacy_docs = list(self.nlp.pipe(docs))
        try:
            tfidf = self.vectorizer.transform(docs)
        except Exception:
            tfidf = None

        results = []
        for i, d in enumerate(spacy_docs):
            ents = [(e.text, e.label_) for e in d.ents]
            pos  = [(t.text, t.pos_) for t in d]
            sent = self.sia.polarity_scores(d.text)

            kws = []
            if tfidf is not None:
                row = tfidf[i].toarray()[0]
                idx = row.argsort()[-top_k_keywords:][::-1]
                kws = [(self.feature_names[j], float(row[j])) for j in idx if row[j] > 0]

            results.append({
                "text": d.text,
                "pos": pos,
                "ner": ents,
                "sentiment": sent,
                "tfidf_keywords": kws
            })
        return results
