#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 28450)
r = np.log(0.5)
t = 5730
y = np.exp((r / t) * x)
plt.yscale("log")
plt.plot(y, 'b')
plt.ylabel('Fraction Remaining')
plt.xlabel('Time (years)')
plt.title('Exponential Decay of C-14')
plt.show()
