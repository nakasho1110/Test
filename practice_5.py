import autograd.numpy as np
from autograd import grad

def tanh(x):
    y = np.exp(-2 * x)
    return (1.0 - y) / (1.0 + y)

from autograd import elementwise_grad as egrad
import matplotlib.pyplot as plt
x = np.linspace(-7, 7, 200)
plt.plot(x, tanh(x),
         x, egrad(tanh)(x),
         x, np.sin(x),
         x, np.cos(x))

# show the plot
plt.show()

