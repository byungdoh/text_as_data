import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
import random
import os

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed) # set hash seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed(42)

# train_x = np.random.uniform(low=0.0, high=10.0, size=500)
# train_noise = np.random.normal(loc=0.0, scale=1, size=500)
# train_y = 0.5*train_x + train_noise

# train_data = np.hstack([np.expand_dims(train_x, axis=1), np.expand_dims(train_y, axis=1)])
# train_data = np.vstack([np.expand_dims(train_x, axis=0), np.expand_dims(train_y, axis=0)])

# dev_x = np.random.uniform(low=-5.0, high=5.0, size=100)
# dev_noise = np.random.normal(loc=0.0, scale=1, size=100)
# dev_y = np.abs(0.5*dev_x) + dev_noise

# dev_data = np.hstack([np.expand_dims(dev_x, axis=1), np.expand_dims(dev_y, axis=1)])
# dev_data = np.vstack([np.expand_dims(dev_x, axis=0), np.expand_dims(dev_y, axis=0)])

def load_np_array(fn):
  arr = np.load(fn, allow_pickle=True)
  return torch.from_numpy(arr[:,0]).float().unsqueeze(-1), torch.from_numpy(arr[:,1]).float().unsqueeze(-1)

train_x, train_y = load_np_array("train_data.npy")
dev_x, dev_y = load_np_array("dev_data.npy")
print(train_x.shape, train_y.shape)
print(dev_x.shape, dev_y.shape)

# np.save("train_data.npy", train_data)
# np.save("dev_data.npy", dev_data)

# fig, ax = plt.subplots()
# ax.scatter(train_x, train_y)
# plt.savefig("train.png")
# plt.clf()

# fig, ax = plt.subplots()
# ax.scatter(dev_x, dev_y)
# plt.savefig("dev.png")
# plt.clf()

print("""
      
      #1-1 is based purely on programming:
      - Take 0.5 points off for each component implemented incorrectly (four components are mentioned in the problem description)
      - Take 0.5 points off if the dev set evaluation is missing
      - Take 0.5 point off if the model definition is incorrect
      - There are many ways to write a training loop, so you may need to spend some time evaluating students' code 
      
      """)

class simple_linear_model(nn.Module):
    def __init__(self):
        super(simple_linear_model, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return(self.linear(x))

model = simple_linear_model()
optimizer = optim.Adam(model.parameters(), lr=1e-2)

# print(model(torch.from_numpy(train_data[:,0]).float().unsqueeze(-1)))

def train(model, inputs, labels, optimizer, criterion):
    model.train()
    # inputs = torch.from_numpy(data[:,0]).float().unsqueeze(-1)
    # labels = torch.from_numpy(data[:,1]).float().unsqueeze(-1)
    optimizer.zero_grad()
    predictions = model(inputs)
    loss = criterion(predictions, labels)
    loss.backward()
    optimizer.step()
    return loss.item()

criterion = nn.MSELoss()
# print(train(model, train_data, optimizer, criterion))
# print(train(model, train_data, optimizer, criterion))
# print(train(model, train_data, optimizer, criterion))
# print(train(model, train_data, optimizer, criterion))
# print(train(model, train_data, optimizer, criterion))

# Evaluation function
def evaluate(model, inputs, labels, criterion):
    model.eval()
    with torch.no_grad():
        # inputs = torch.from_numpy(data[:,0]).float().unsqueeze(-1)
        # labels = torch.from_numpy(data[:,1]).float().unsqueeze(-1)
        predictions = model(inputs)
        loss = criterion(predictions, labels)

    return loss.item()

for i in range(300):
    train_loss = train(model, train_x, train_y, optimizer, criterion)
    dev_loss = evaluate(model, dev_x, dev_y, criterion)
    print(train_loss, dev_loss)

print(model.linear.weight)
print(model.linear.bias)

print("""
      
      #1-2 should be something like y = 0.4913x + 0.0989 (be generous with rounding)
      The code simply involves calling model.linear.weight and model.linear.bias
      
      """)

print("""
      
      #1-3: Throughout the 300 epochs, the dev loss initially decreases but starts to increase.
      
      """)

print("""
      
      #1-4: I mentioned in the problem description that "overfitting" is not exactly the answer we're looking for.
      The answer I'm looking for is something like "the train and dev data come from different distributions."
      train.png and dev.png visualizes the two datasets, which might help you get an idea of the expected answer.
      This answer can have many different forms, such as "the range of the input feature is different."
      -0.5 point if the answer just says "overfitting" (because it's actually technically true), -1 point if the answer is completely off.

      """)