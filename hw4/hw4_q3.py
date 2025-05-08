import transformers
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# import GPT-2 pretrained model
gpt = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

sents = ["A bird is a",
         "A bird is not a",
         "A flower is a",
         "A flower is not a",
         "A hammer is a",
         "A hammer is not a"]

for i in sents:
    gpt_input = tokenizer(i, return_tensors="pt")
    # print(tokenizer.convert_ids_to_tokens(gpt_input.input_ids.squeeze()))
    outputs = gpt(**gpt_input)
    logits = outputs.logits.squeeze()
    idx = torch.topk(logits[-1], 5).indices
    print(tokenizer.convert_ids_to_tokens(idx))

print("""
      
      #3-1: The answers should be printed above.
      If the code is incorrect, the answers will also be incorrect,
      so just take -2 points off all together if the answers severely deviate from what is printed above.
      The answer I'm looking for as the unnatural continuations are sentences like "A bird is not a bird".
      My guess as to why this happens is that GPT-2 blindly 'copies' earlier words to predict the next word. 
      (but do give full credit for any reasonable guess)
      
      """)

# # notice the trailing space of the second sequence
# sents = ["In the kingdom of the blind, the one",
#          "In the kingdom of the blind, the one "]

# notice the trailing space of the second sequence
sents = ["Birds of a feather",
         "Birds of a feather "]

for i in sents:
    gpt_input = tokenizer(i, return_tensors="pt")
    # print(tokenizer.convert_ids_to_tokens(gpt_input.input_ids.squeeze()))
    outputs = gpt.generate(**gpt_input, max_new_tokens=5)
    print(tokenizer.decode(outputs[0], skip_special_tokens=False))


print("""
      
      #3-2: The answers should be printed above.
      If the code is incorrect, the answers will also be incorrect,
      so just take -1 point off all together if the answers severely deviate from what is printed above.
      One preprocessing method could be to simply remove the trailing whitespace,
      with .strip() or something like that.

      """)