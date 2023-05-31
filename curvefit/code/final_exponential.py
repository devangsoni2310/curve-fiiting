# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 10:43:14 2023

@author: User
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Define the exponential function
def exponential(x, a, b, c):
    return a * np.exp(-b * x) + c

# Generate some noisy data
x_data = np.linspace(0, 10, 50)
y_data = exponential(x_data, 10, 0.5, 2)
y_data = y_data + 0.5 * np.random.normal(size=len(x_data))

# Fit the data using curve_fit
popt, pcov = curve_fit(exponential, x_data, y_data)

# The optimized parameters
# print('Optimized parameters:', popt)
a, b, c = popt


# Plot the data and fitted curve
x_model= np.linspace(min(x_data),max(x_data),50)
y_model= exponential(x_data, a, b, c)
plt.scatter(x_data, y_data, label='Original Data')
plt.plot(x_model, y_model , 'r-', label='Fit')
plt.xlabel('time ($sec$)')
plt.ylabel('velocity ($m/sec$)')
plt.title('Exponential curve fitting')
plt.grid('on')
plt.legend()
plt.show()

residual=y_data-y_model
plt.plot(x_model, residual)
plt.title('Residual exponential fit')
plt.ylabel('difference)')

plt.grid('on')
plt.show()