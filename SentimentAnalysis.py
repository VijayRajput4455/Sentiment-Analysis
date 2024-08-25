import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import pickle

# Ensure necessary NLTK resources are downloaded
nltk.download('stopwords')
nltk.download('punkt')

class SentimentAnalysisModel:
    def __init__(self, vector_model_path, model_path):
        self.vector_model = self.load_model(vector_model_path)
        self.model = self.load_model(model_path)
        self.ps = PorterStemmer()
        self.stop_words = set(stopwords.words("english"))
        self.punctuation_set = set(string.punctuation)

    def load_model(self, file_path):
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except (FileNotFoundError, pickle.PickleError) as e:
            raise RuntimeError(f"Error loading model from {file_path}: {str(e)}")

    def transform_text(self, text):
        # Convert text to lowercase
        text = text.lower()

        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)

        # Tokenize text and filter out non-alphanumeric tokens
        tokens = [word for word in nltk.word_tokenize(text) if word.isalnum()]

        # Remove stopwords and punctuation, then apply stemming
        processed_text = [
            self.ps.stem(word) for word in tokens if word not in self.stop_words and word not in self.punctuation_set
        ]

        # Join the processed words back into a single string
        return " ".join(processed_text)

    def predict_sentiment(self, text):
        try:
            processed_text = self.transform_text(text)
            vectors = self.vector_model.transform([processed_text])
            result = self.model.predict(vectors)[0]
            response = str("Positive" if result == 1 else "Negative")
            return response
        except Exception as e:
            response = str(f"An error occurred: {str(e)}")
            return response

# Example usage
# vector_model_path = "Pickles_Files\CountVectorizer.pkl"
# model_path = "Pickles_Files\Model.pkl"

# sentiment_model = SentimentAnalysisModel(vector_model_path, model_path)
# text = "I love this product!"
# print(sentiment_model.predict_sentiment(text))
