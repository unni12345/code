import torch
import math

dtype = torch.float
device = torch.device("cpu")

x = torch.linspace(-math.pi,math.pi, 2000)
y = torch.sin(x)

a = torch.randn((), device=device, dtype=dtype)
b = torch.randn((), device=device, dtype=dtype)
c = torch.randn((), device=device, dtype=dtype)
d = torch.randn((), device=device, dtype=dtype)

learning_rate = 1e-6

# print y
# print(y)

for t in range(2000):
    # forward pass calculate y_pred = a + bx + cx^2 + dx^3
    y_pred = a + b*x + c*x**2 + d*x**3
    
    # compute and print loss
    loss = (y_pred - y).pow(2).sum().item()
    if t % 100 == 99:
        print(t, loss)

    # backpropagation to compute the gradients of a,b,c,d w.r.t loss
    # derivative of squared loss function
    grad_y_pred = 2.0*(y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x**2).sum()
    grad_d = (grad_y_pred * x**3).sum()

    # Update weights
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d

print(f'Result: y = {a} + {b} x + {c} x^2 + {d} x^3')