from d2l import torch as d2l
import torch

def h(x):
    result = []
    for i in x:
        result.append(torch.sin(i))
    return result

def hh(x):
    result = []
    for i in x:
        i.requires_grad_(True)
        y = torch.sin(i)
        y.backward()
        result.append(i.grad.clone())
        i.grad.zero_()
    return result

x = torch.arange(0, 7, 0.1)
d2l.plot(x, [h(x), hh(x)], 'x', 'f(x)', legend=['sin(x)', 'sin\'(x)'])