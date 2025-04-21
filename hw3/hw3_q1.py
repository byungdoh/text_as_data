import os, sys
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

print("""
      
      #1-1 should be fairly straightforward:
      - Take 0.5 points off only once for 1-1 if the counts are different (because 1-2 and 1-3 depend on it)
      
      """)

# from https://stackoverflow.com/questions/50060241/how-to-use-glove-word-embeddings-file-on-google-colaboratory
embeddings_index = {}
with open("glove.6B.50d.verbs.txt", "r+") as f:
    for line in f:
        values = line.strip().split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embeddings_index[word] = coefs

# print(embeddings_index.keys())
X = np.vstack(list(embeddings_index.values()))
kmeans = KMeans(n_clusters=5, random_state=0, n_init="auto").fit(X)

print(Counter(list(kmeans.labels_)))
for i, j in zip(list(embeddings_index.keys()), list(kmeans.labels_)):
    print(i, j)

print("""
      
      The answer I'm looking for #1-2 is "yes" (for both synonyms and antonyms):
      - Take 0.5 points off if the answer is incorrect
      - Take 0.5 points off for each example pair not mentioned
        (-0.5 if synonym pair missing, -0.5 if antonym pair missing, -1 if both missing)
      
      """)


print("""
      
      #1-3 should be straightforward:
      - Be generous with rounding/floating point precision etc.
      - But take 0.5 points off if the output is grossly incorrect or the code does something wrong
      
      """)

# Step 2: Perform PCA
num_components = 2  # Choose the desired number of components
pca = PCA(n_components=num_components)
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)
pca_result = pca.fit_transform(X_standardized)
cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)

# print(pca_result)
print(cumulative_explained_variance)
print("The response should be 21.62% or so")

print("""
      
      The output of #1-4 should look something like 1-4.png in the directory:
      - Take 0.5 points off if the output is grossly incorrect or the code does something wrong
      
      """)

# Example code
# fig, ax = plt.subplots()

# for i, j in zip(list(embeddings_index.keys()), pca_result):
#     ax.scatter(j[0], j[1])
#     ax.annotate(i, (j[0], j[1]), fontsize=6)

# fig.savefig("1-4.png")

print("""
      
      #1-5 should be straightforward:
      - Take 1 point off for each pair missing
      - Be generous regarding what it means to be related in meaning. Example pairs based on the figure can include:
       (listen, hear), (wish, want), (think, understand), (boil, bake), ...
      
      """)