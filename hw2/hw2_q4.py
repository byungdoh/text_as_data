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
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline


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

print("""
      
      #4 might also be somewhat tricky due to preprocessing (number removal using regex).
      So given that all the requested preprocessing is implemented (-0.5 points if not):

      - Don't take points off 4-2 if the accuracy scores are within a similar ballpark range of what I have
      - Don't take points off 4-3 if students get at least 3 out of 5 words on both positive class and negative class (-0.5 points otherwise)
      - Take 0.5 points off 4-3 if students flipped the features (i.e. answered positive class words with negative class words and vice-versa)
      - Be generous with rounding/floating point precision etc.

      """)


print("4-1")
print("""
      Something about dictionaries not being able to apply across domains (while NB/SVM is data driven).
      For example, a dictionary developed with Amazon product reviews is not going to be applicable to hotel reviews.
      """)

# 4-2 code

df = pd.read_csv(sys.argv[1])[:1000]
df["label"] = (df["stars"] > 4).astype(int)

clean_text = []
# stars = []
for _, i in df.iterrows():
    clean_text.append(preprocess(i["text"]))
    # stars.append(i["stars"])

df["clean_text"] = clean_text
# df["label"] = [-1 if i > 4 else 1 for i in stars]
# print(df)

# partition data (done for you for consistency)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

index = kf.split(df["clean_text"])

scores = []

# Define the svm_classifier_rbf within the loop to make it available
for _, (train_index, test_index) in enumerate(index):
    # Define the pipeline within the loop:
    svm_classifier_l = Pipeline([
        ("freq", CountVectorizer()),  # Convert text data to TF-IDF features
        ("svm", SVC(kernel="linear"))  # Linear SVM classifier
    ])
    svm_classifier_l.fit(df["clean_text"][train_index], df["label"][train_index])
    y_pred = svm_classifier_l.predict(df["clean_text"][test_index])
    accuracy = accuracy_score(df["label"][test_index], y_pred)
    scores.append(accuracy)

print("4-2")
print(scores)

# 4-3 code

index = kf.split(df["clean_text"])
for i, (train_index, test_index) in enumerate(index):
    # Define the pipeline within the loop:
    if i == 0:
      svm_classifier_l = Pipeline([
          ("freq", CountVectorizer()),  # Convert text data to TF-IDF features
          ("svm", SVC(kernel="linear"))  # Linear SVM classifier
      ])
      svm_classifier_l.fit(df["clean_text"][train_index], df["label"][train_index])
      # print(svm_classifier_l["freq"])
      weights = svm_classifier_l["svm"].coef_.toarray().flatten()
      idx = np.argsort(weights)
      # print(weights[idx[-5:]])
      # print(idx[-5:])
      # print(weights[idx[:5]])
      # print(idx[:5])
      vectorizer = CountVectorizer()
      X = vectorizer.fit_transform(df["clean_text"])
      # print(X.toarray())
      features = vectorizer.get_feature_names_out()
      pos_features = features[idx[-5:]]
      neg_features = features[idx[:5]]
      print("4-3")
      print("== top 5 words predictive of pos class == ")
      print(pos_features)
      print("== top 5 words predictive of neg class == ")
      print(neg_features)

# for i in list(neg_features) + list(pos_features):
#     labels = []
#     for _, j in df.iterrows():
#         if i in j["clean_text"].split(" "):
#             labels.append(j["label"])
#     print(i, sum(labels)/len(labels))

print("""
      
      #5 is related to the final project, and I ask you to use your best judgment to evaluate the responses.
      It's probably way too close to the end of the semester to request major changes to their projects,
      but feel free to deduct 0.5 points or so if the responses are extremely unclear and hard to understand
      (as clear writing is one of the goals of this course).

      """)
