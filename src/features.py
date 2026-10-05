from sklearn.feature_extraction.text import TfidfVectorizer

class TfidfFeatureExtractor:
        def __init__(self) -> None:
                self.vectorizer = TfidfVectorizer(max_features=10000)

        def transform(self, text):
                return self.vectorizer.transform(text)
                

        def fit_transform(self, text):
                return self.vectorizer.fit_transform(text)

        

        

        


