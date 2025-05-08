import transformers
import torch
import torch.nn as nn
from transformers import BertForMaskedLM, BertTokenizerFast

# import BERT-base pretrained model
bert = BertForMaskedLM.from_pretrained('bert-base-uncased')

# BERT is special, and optimized by huggingface
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

bert_input = tokenizer("Hello, my [MASK] is cute", return_tensors="pt")
# print(tokenizer.convert_ids_to_tokens(bert_input.input_ids.squeeze()))
outputs = bert(**bert_input)
logits = outputs.logits.squeeze()

idx = torch.topk(logits[4], 5).indices

print(tokenizer.convert_ids_to_tokens(idx))

print("""
      
      #2-1: The top five predictions are those printed above.
      The answer I'm looking for as the unnatural prediction is "name".
      The most likely reason for this prediction is that "Hello, my name is ..." is an extremely frequent sequence.
      If the code is incorrect, the answers will also be incorrect,
      so just take -2 points off all together if the answers severely deviate from what is printed above.

      """)

sents = ["The [MASK] works as a pilot.",
         "The [MASK] works as a nurse.",
         "The [MASK] works as a engineer.",
         "The [MASK] works as a housekeeper."]

man = 2158
woman = 2450

for i in sents:
    bert_input = tokenizer(i, return_tensors="pt")
    print(tokenizer.convert_ids_to_tokens(bert_input.input_ids.squeeze()))
    bert.eval()
    with torch.no_grad():
        outputs = bert(**bert_input)
    logits = outputs.logits.squeeze()
    sm = nn.Softmax(dim=-1)
    probs = sm(logits)
    print("man", probs[2][man])
    print("woman", probs[2][woman])

    idx = torch.topk(probs[2], 5).indices

    # woman: 2450
    # man: 2158

    print(tokenizer.convert_ids_to_tokens(idx))
    # print(idx)


print("""
      
      #2-2: The probabilities should be those printed above (again, generous with rounding).
      If the code is incorrect, the answers will also be incorrect,
      so just take -2 points off all together if the answers severely deviate from what is printed above.
      For the last question, I think both sides make sense and should be given full credit:
      One side might say this is bad because it propogates gender stereotypes related to professions.
      But the other side might say this is okay because it accurately reflects facts about the world (e.g. there are more female nurses than male nurses).

      """)
