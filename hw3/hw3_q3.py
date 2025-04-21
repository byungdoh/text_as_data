import torch

print("""
      
      #3-1 and #3-2 are very straightforward, -0.5 for each incorrect answer.
      
     """)

a = torch.tensor([2.], requires_grad=True)
b = torch.tensor([3.], requires_grad=True)
c = a+b
d = b**2
e = c*d

print(e)
e.backward()
print("Answer to #3-1:", b.grad)

print("See 3-2.jpeg for answers to #3-2.")

print("""
      
      #4 is related to the final project, and I ask you to use your best judgment to evaluate the responses.
      
      It's probably way too close to the end of the semester to request major changes to their projects,
      but feel free to deduct 1 point or so if the visualization in #4-2 is extremely unclear (if the axes are missing or such),
      0.5 points or so if the response to #4-3 does not match the visualization or is extremely hard to understand
      (as clear writing is one of the goals of this course).

      """)