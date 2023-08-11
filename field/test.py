import numpy as np
import matplotlib.pyplot as plt
from mylib.field.sigmoid import sigmoid

x = np.linspace(0,100,10000)
y = sigmoid(x, 100, 0.2, x0=50, h=20)

plt.plot(x,y)
plt.show()