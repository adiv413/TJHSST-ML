import numpy as np
import matplotlib.pyplot as plt
# make your plot outputs appear and be stored within the notebook

x = np.linspace(0,20, 100)
plt.plot(x, np.sin(x))

plt.xlabel("x")
plt.ylabel("sin(x)")
plt.title("A sine curve")
plt.show()