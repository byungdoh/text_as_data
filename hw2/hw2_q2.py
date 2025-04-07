import pandas as pd
import sys
import string
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

def load_dict(fn):
    my_set = set()
    with open(fn, "r+") as f:
        lines = f.readlines()
        for line in lines:
            my_set.add(line.strip())
    return my_set

def preprocess(line):
    return line.strip().lower().translate(str.maketrans('', '', string.punctuation)).split(" ")

print("""
      
      #2 should be fairly straightforward:
      - Take 0.5 points off only once for 2-1 if the counts are different (because 2-3 depends on it)
      - Take 0.5 points off only once for 2-2 if the numbers are different (because 2-3 could depend on it)
      - Take 0.5 points off for 2-3 if the deterministic classification rule was applied incorrectly
      - Don't take points off if students report macro average (although weighted/micro avg was intended)
      - Be generous with rounding/floating point precision etc.
      
      """)

# code for 2-1
print("2-1")
df = pd.read_csv(sys.argv[1])
print(df["stars"].median())
df["label"] = (df["stars"] > 4).astype(int)
print(len(df[df["label"] == 1])/7500)
print(len(df[df["label"] == 0])/7500)

# code for 2-2
pos_words = load_dict(sys.argv[2])
neg_words = load_dict(sys.argv[3])

all_scores = []
for _, i in df.iterrows():
    sent_score = 0
    for word in preprocess(i["text"]):
        if word in pos_words:
            sent_score += 1
        elif word in neg_words:
            sent_score -= 1
    all_scores.append(sent_score)

print("2-2")
print(np.mean(all_scores))
print(np.std(all_scores))

# code for 2-3
print("2-3")
all_preds = [1 if i > 0 else 0 for i in all_scores]
all_targets = list(df["label"])

# print(accuracy_score(all_targets, all_preds))
print(classification_report(all_targets, all_preds))

# code for 2-4
print("2-4")
print("""
      
      The answer I'm looking for is something like:
      "The example was classified incorrectly as positive/negative because it includes the positive/negative word X, but this is irrelevant"
      "The reviews only contain the first few sentences and are not complete (they end with '... More')"
      
      - Take 0.5 points off if the reason for the misclassification of the 1-star review doesn't explicitly mention the positive dictionary words
      - For the misclassification of 5-star reviews, it's fine to say something like "the review doesn't contain any dictionary words"
      (therefore resulting in a score of 0 and classified as negative)
      
      """)


print("printing just one example")
for h, i in df.iterrows():
    if all_preds[h] >= 1 and i["stars"] == 1:
        print("== misclassification of 1-star review ==")
        print(i["text"])
        for j in preprocess(i["text"]):
            if j in pos_words:
                print("    " + j)
        break

for h, i in df.iterrows():
    if all_preds[h] <= 0 and i["stars"] == 5:
        print("== misclassification of 5-star review ==")
        print(i["text"])
        for j in preprocess(i["text"]):
            if j in neg_words:
                print("    " + j)
        break
