import pandas as pd
import sys
import string
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nltk.corpus import stopwords
import re

print("""
      
      #3 might be somewhat tricky due to preprocessing (number removal using regex).
      So given that all the requested preprocessing is implemented (-0.5 points if not):

      - Don't take points off if the confusion matrix + classification report are within a similar ballpark range of what I have
      - Don't take points off if students report macro average (although weighted/micro avg was intended)
      - Be generous with rounding/floating point precision etc.
      
      """)

def load_dict(fn):
    my_set = set()
    with open(fn, "r+") as f:
        lines = f.readlines()
        for line in lines:
            my_set.add(line.strip())
    return my_set

def preprocess(line):
    j = " ".join([i for i in line.strip().lower().translate(str.maketrans('', '', string.punctuation)).split(" ") if i not in stopwords.words("english")])
    return re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", j)

# code for 3-1
print("3-1")
df = pd.read_csv(sys.argv[1])
df["label"] = (df["stars"] > 4).astype(int)

clean_text = []
for _, i in df.iterrows():
    clean_text.append(preprocess(i["text"]))

df["clean_text"] = clean_text
# print(df)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Separate features (X) and labels (y)
X_train, y_train = train_df['clean_text'], train_df['label']
X_test, y_test = test_df['clean_text'], test_df['label']

# Initialize CountVectorizer and Naive Bayes classifier
vectorizer = CountVectorizer()
nb_classifier = MultinomialNB()

# Fit and transform the training data
X_train_vectorized = vectorizer.fit_transform(X_train)
# Transform the test data using the same vectorizer
X_test_vectorized = vectorizer.transform(X_test)

# Train the classifier on the training data
nb_classifier.fit(X_train_vectorized, y_train)
# Predict on the test data
y_pred_nb = nb_classifier.predict(X_test_vectorized)

# Display confusion matrix
print("\nConfusion Matrix:")
conf_mat = confusion_matrix(y_test, y_pred_nb)
print(conf_mat)

# Display classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_nb))

print("3-2")
print("""
      
      Some larger linguistic unit like bigram/trigrams seems like the most straightforward answer.
      The point of this question is to get students to think outside the box.
      So as long as they try to provide a reasonable justification, give full 2 points.

      """)
