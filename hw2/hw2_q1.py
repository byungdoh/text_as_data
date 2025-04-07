import sys
import string

print("""
      
      #1 should be fairly straightforward:
      - Take 0.5 points off only once for 1-1 if the counts are different (because 1-2 and 1-3 depend on it)
      - Take 0.5 points off for code in 1-1 if the preprocessing isn't done as directed or if the counting is somehow off
      - Be generous with rounding/floating point precision etc.
      
      """)

# code for 1-1
def preprocess(fn):
    all_sents = []
    with open(fn, "r+") as f:
        lines = f.readlines()
        for line in lines:
            all_sents.append(line.strip().lower().translate(str.maketrans('', '', string.punctuation)).split(" "))
    return all_sents

corpus = preprocess(sys.argv[1])

a1 = 0
a2 = 0
a3 = 0
a4 = 0

for sent in corpus:
    for idx in range(len(sent)-1):
        if sent[idx] == "united" and sent[idx+1] == "nations":
            a1 += 1
        elif sent[idx] == "united" and sent[idx+1] != "nations":
            a2 += 1
        elif sent[idx] != "united" and sent[idx+1] == "nations":
            a3 += 1
        elif sent[idx] != "united" and sent[idx+1] != "nations":
            a4 += 1
        else:
            raise ValueError


print("1-1")
print(a1, a2, a3, a4)

# code for 1-2
denom = a1 + a2 + a3 + a4
print("1-2")
print(((a1+a2)*(a1+a3))/(denom))

# code for 1-3
e1 = ((a1+a2)*(a1+a3))/(denom)
e2 = ((a1+a2)*(a2+a4))/(denom)
e3 = ((a1+a3)*(a3+a4))/(denom)
e4 = ((a3+a4)*(a2+a4))/(denom)
chisq = (((a1-e1)**2)/e1) + (((a2-e2)**2)/e2) + (((a3-e3)**2)/e3) + (((a4-e4)**2)/e4)
print("1-3")
print(chisq)
