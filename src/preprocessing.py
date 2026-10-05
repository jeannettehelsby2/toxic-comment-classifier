import string
import nltk
from nltk.corpus import stopwords

class TextPreprocessor:
    """
    Preprocess text comments for the Toxic Comment Classification model.

    The preprocessor:
    - converts text to lowercase
    - tokenises text
    - removes English stop words
    - removes punctuation
    - removes empty tokens
    """
    def __init__(self) -> None:
        """Init method."""

        # Download the NLTK english word-stop dataset.
        nltk.download("stopwords", quiet=True)

        #Store stop words for usage duroing text cleanning
        self.stop_words = set(stopwords.words("english"))

        #Create transaltion table that removes puntuation.
        self.punctuation_table = str.maketrans("","",string.punctuation,) 

    def clean(self, text: str) -> str:
        """
        Clean a single comment.

        Args:
            text: Raw comment text.

        Returns:
            Cleaned lowercase text with stop words
            and punctuation removed.
        """

        # Convert the text to lowercase and split it into words.
        words = text.lower().split()
        
        # Remove stop words and punctuation from each word.
        cleaned_words = [word.translate(self.punctuation_table)
                         for word in words
                         if word not in self.stop_words
        ]
        
        # Remove empty words and join the words back into a string.
        return " ".join (word for word in cleaned_words if word)