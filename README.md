
# Sentiment Analysis
This project involves the development of a sentiment analysis classifier using various machine learning techniques on the Large Movie Review Dataset. The goal is to accurately classify movie reviews as positive or negative, leveraging different models to compare their performance and efficacy.

## Data Set
The dataset used in this project is sourced from IMDB and consists of 50,000 movie reviews. It is evenly divided into 25,000 training reviews and 25,000 testing reviews, with balanced sentiment labels—half of the reviews are positive, and the other half are negative.

Data Preprocessing
To prepare the data for effective sentiment analysis, the following preprocessing steps were undertaken:

- Stopword Removal: Commonly used words that do not contribute to the sentiment (e.g., "and," "the," "is") were removed to focus on the more meaningful terms.
- Lowercasing: All text data was converted to lowercase to ensure uniformity and avoid treating words with different cases as separate entities.
- Punctuation Removal: Punctuation marks were removed to clean the text and reduce noise.
- Tokenization: The text was tokenized, breaking down sentences into individual words or tokens to facilitate analysis.

## Models
- [CountVectorizer.pkl](https://drive.google.com/file/d/1Qz6i7-dl74BuVBgJhLAnH-WDhufIyPGk/view?usp=sharing)
- [Model.pkl](https://drive.google.com/file/d/1CfGdz9e6MiRuy1SAkKynbA2YJ6viWaxU/view?usp=sharing)