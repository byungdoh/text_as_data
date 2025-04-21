import pandas as pd
import nltk
from collections import defaultdict
from nltk.corpus import stopwords
import re
import string

from pprint import pprint
import gensim
import gensim.corpora as corpora
from gensim.models.ldamodel import LdaModel
from gensim.corpora import Dictionary
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary

print("""
      
      #2-1 should be straightforward, -0.5 points for each incorrect answer.
      
      """)

# nltk.download('stopwords')
biden_dict = defaultdict(list)
trump_dict = defaultdict(list)
df = pd.read_csv("election_tweets.csv")
for _, i in df.iterrows():
    # print(i["hashtag"])
    if i["hashtag"] == "joebiden":
        biden_dict[i["date"]].append(i["tweet"])
    elif i["hashtag"] == "donaldtrump":
        trump_dict[i["date"]].append(i["tweet"])
    else:
        print(i)
        raise ValueError

print("Biden", len(biden_dict.keys()))
print("Trump", len(trump_dict.keys()))

print("""
      
      #2-2 might be somewhat tricky due to different ways to implement preprocessing (e.g. number removal using regex).
      So just check the code to see that all the requested preprocessing is implemented (-0.5 points per each item if not).
      
      """)

sw = stopwords.words("english")
# re.sub(r'[^\x00-\x7f]',r'', your-non-ascii-string) 
# print(df)

biden_list = []
trump_list = []

for i in biden_dict:
    processed_words = []
    curr_str = " ".join(biden_dict[i])
    curr_str = re.sub(r'\n', r' ', curr_str).lower()
    ascii_str = re.sub(r'[^\x00-\x7f]', r'', curr_str)
    for j in ascii_str.split():
        if (j not in sw) and (not j.startswith("@")) and (not j.startswith("http")):
            k = j.translate(str.maketrans('', '', string.punctuation))
            if k != "":
                processed_words.append(k)
    biden_list.append(processed_words)

for i in trump_dict:
    processed_words = []
    curr_str = " ".join(trump_dict[i])
    curr_str = re.sub(r'\n', r' ', curr_str).lower()
    ascii_str = re.sub(r'[^\x00-\x7f]', r'', curr_str)
    for j in ascii_str.split():
        if (j not in sw) and (not j.startswith("@")) and (not j.startswith("http")):
            k = j.translate(str.maketrans('', '', string.punctuation))
            if k != "":
                processed_words.append(k)
    trump_list.append(processed_words)

print("""
      
      #2-3 should be pretty straightforward, and should yield output that looks something like the following
      But do be generous to minor difference that may arise due to difference in preprocessing
      
      """)

preprocessed = trump_list + biden_list
common_dictionary = Dictionary(preprocessed)

# Convert preprocessed corpus to bag of words
corpus = [common_dictionary.doc2bow(text) for text in preprocessed]
seed = 42
lda = LdaModel(corpus, id2word=common_dictionary, num_topics=10,
               alpha='auto', eta='auto', random_state=42, passes=5, iterations=1000)

pprint(lda.print_topics())

print("""
      
      The answers I'm looking for in #2-4 are:
      - 1) No, there is not a lot of variety in the two topics
      - 2) The "reasonable guess" I'm looking for is that "trump" and "biden" are too frequent in the twitter data
           (do give full credit if other reasons make sense, -0.5 if it makes absolutely no sense)
      
      """)

print("""
      
      For #2-5:
      - 1) The output should look something like below (again, be generous to minor differences)
      - 2) The code should be basically a copy-paste of what's above, with "trump" and "biden" filtered out
      - 3) There is one topic that seems to be about the debate, and a set of others that have to do with the election
           (full credit as long as they make a reasonable attempt)
      
      """)

biden_list = []
trump_list = []

for i in biden_dict:
    processed_words = []
    curr_str = " ".join(biden_dict[i])
    curr_str = re.sub(r'\n', r' ', curr_str).lower()
    ascii_str = re.sub(r'[^\x00-\x7f]', r'', curr_str)
    for j in ascii_str.split():
        if (j not in sw) and (not j.startswith("@")) and (not j.startswith("http")) and (not "trump" in j) and (not "biden" in j):
            k = j.translate(str.maketrans('', '', string.punctuation))
            if k != "":
                processed_words.append(k)
    biden_list.append(processed_words)

for i in trump_dict:
    processed_words = []
    curr_str = " ".join(trump_dict[i])
    curr_str = re.sub(r'\n', r' ', curr_str).lower()
    ascii_str = re.sub(r'[^\x00-\x7f]', r'', curr_str)
    for j in ascii_str.split():
        if (j not in sw) and (not j.startswith("@")) and (not j.startswith("http")) and (not "trump" in j) and (not "biden" in j):
            k = j.translate(str.maketrans('', '', string.punctuation))
            if k != "":
                processed_words.append(k)
    trump_list.append(processed_words)

preprocessed = trump_list + biden_list
common_dictionary = Dictionary(preprocessed)

# Convert preprocessed corpus to bag of words
corpus = [common_dictionary.doc2bow(text) for text in preprocessed]
seed = 42
lda = LdaModel(corpus, id2word=common_dictionary, num_topics=10,
               alpha='auto', eta='auto', random_state=42, passes=5, iterations=1000)

pprint(lda.print_topics())