import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec

def clean_text(text):
    """
    Cleans a single comment by lowercasing and removing punctuation.
    @param text: raw comment string
    @return: cleaned lowercase string with punctuation removed
    """
    #toxic_words = toxic_text.lower().split()
    #toxic_words_clean = [word.translate(str.maketrans('', '', string.punctuation)) for word in toxic_words if word not in stop_words and word != '']
    #toxic_words_clean = [word for word in toxic_words_clean if word != '']

    return text.lower().translate(str.maketrans('', '', string.punctuation))

   

def get_features(text_train, text_test,  method='tfidf'):
    """
    TO_DO
    """

    #text → TfidfVectorizer → sparse matrix of 
    if method == 'tfidf':
        vectorizer = TfidfVectorizer(max_features=10000)
        X_train = vectorizer.fit_transform(text_train)
        X_test = vectorizer.transform(text_test)
        return X_train, X_test


    #text → train Word2Vec → average word vectors → array of numbers
    if method == 'word2vec':
        sentences = [comment.lower().split() for comment in text_train]
        w2v_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
        X_train = np.array([_get_comment_vector(c, w2v_model, 100) for c in text_train])
        X_test = np.array([_get_comment_vector(c, w2v_model, 100) for c in text_test])
        return X_train, X_test
    raise ValueError(f"Unknown feature method: {method}")

def _gte_commnet_vector(comment, w2w_model, vector_size):
    """
    Converts a single commnet into averaged Word2Vec vector
    @param comment: Raw text for a single comment
    @return: a 100 vector if no words were recognized
    """
    
    # split comment into words
    words = comment.lower().split()
          
    # get vector for each word
    vectors = [ w2v_model.wv[word] for word in words if word in w2v_model.wv]
       
    # average all vectors
    if len (vectors) == 0:
        return np.zeros(vector_size)
        
    # return the average vector
    return np.mean (vectors, axis=0)


