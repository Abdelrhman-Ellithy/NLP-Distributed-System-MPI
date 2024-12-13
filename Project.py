from mpi4py import MPI
import numpy as np
import pandas as pd
import tensorflow as tf
import re
import emoji
import string
from nltk.corpus import stopwords
import os
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Suppress TensorFlow logs and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore")
StopWords = stopwords.words("english")
exclude = string.punctuation
def remove_stopwords(text):
    return ' '.join(word for word in text.split() if word.lower() not in StopWords)

def remove_punc(text):
    for char in exclude:
        text = text.replace(char, '')
    return text

def remove_digits(text):
    return re.sub(r'\d+', '', text)

def remove_html_tags(text):
    pattern = re.compile('<.*?>')
    return pattern.sub('', text)

def remove_url(text):
    pattern = re.compile(r'https?://\S+|www\.\S+')
    return pattern.sub(r'', text)

def preprocess(data):
    data.dropna(inplace=True)
    data['Sentiment'] = data['Sentiment'].astype(str).apply(remove_html_tags)
    data['Sentiment'] = data['Sentiment'].apply(remove_url)
    data['Sentiment'] = data['Sentiment'].apply(emoji.demojize)
    data['Sentiment'] = data['Sentiment'].apply(remove_digits)
    data['Sentiment'] = data['Sentiment'].apply(remove_punc)
    data['Sentiment'] = data['Sentiment'].apply(remove_stopwords)
    return data

comm = MPI.COMM_WORLD
rank = comm.Get_rank() 
size = comm.Get_size()  

if rank == 0:
    print("Loading dataset...")
    train_df = pd.read_csv('twitter_training.csv')
    df = train_df.drop(columns=train_df.columns[:2])
    print("Dataset shape:", df.shape)
    df = df[[df.columns[1], df.columns[0]]]
    df = df.rename(columns={df.columns[0]: 'Sentiment', df.columns[1]: 'Analysis'})
    data_split = np.array_split(df, size)  # Split data across processes
else:
    data_split = None

local_data = comm.scatter(data_split, root=0)
local_preprocessed = preprocess(local_data)

gathered_data = comm.gather(local_preprocessed, root=0)

if rank == 0:
    preprocessed_data = pd.concat(gathered_data, ignore_index=True)
    print("Preprocessing complete. Preprocessed data shape:", preprocessed_data.shape)
    X = preprocessed_data.drop('Analysis', axis=1)
    y = preprocessed_data['Analysis']
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    print("Train/Test shapes:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    vectorizer = CountVectorizer(
        lowercase=True,
        max_features=10000,
        ngram_range=(1, 2)
    )
    X_train_vec = vectorizer.fit_transform(X_train['Sentiment'])
    X_test_vec = vectorizer.transform(X_test['Sentiment'])
    vocab = vectorizer.get_feature_names_out()
    print("Vocabulary size:", len(vocab))

    tree_classifier = DecisionTreeClassifier(criterion='entropy', random_state=44)
    tree_classifier.fit(X_train_vec, y_train)

    y_pred = tree_classifier.predict(X_test_vec)
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", conf_matrix)

    report = classification_report(y_test, y_pred)
    print("Classification Report:\n", report)

else:
    preprocessed_data = None
