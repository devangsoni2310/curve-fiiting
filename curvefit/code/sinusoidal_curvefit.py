# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 14:57:22 2023

@author: User
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Define a function to generate the sample data
def sinusoidal(x, a, b, c, d):
    return a * np.sin(b * x + c) + d

# Generate some sample data
x_data = np.linspace(0, 48, 96)
y_data = sinusoidal(x_data, 5, np.pi/12, 3*np.pi/2, 10)
y_noise = 0.75 * np.random.normal(size=y_data.size)
y_data = y_data + y_noise



# Fit the data to the function using curve_fit
# initial guess p0=[4, 2, np.pi/2, 15]
popt, pcov = curve_fit(sinusoidal, x_data, y_data,p0=[5, np.pi/12, 3*np.pi/2, 10])

# The optimized parameters
a, b, c, d = popt


x_model = np.linspace(min(x_data),max(x_data),100)
y_model = sinusoidal(x_data, a, b, c, d)

# Calculate Residuals
r = y_data - y_model


# plot data, fitted curve and residual
plt.scatter(x_data,y_data,  label='Original Data')
plt.plot(x_data, y_model,color='r', label='Fit')
plt.plot(x_data, r)
plt.xlabel('Time of the day(2 days 48 hours)')
plt.ylabel('Temperature ($^\circ$C)')
plt.title('Sinusoidal curve fitting')
plt.grid('on')
plt.legend()
plt.show()
plt.figure
plt.plot(x_data,r)
plt.grid('on')
plt.figure()
plt.scatter(x_model, y_model)
plt.show('on')

# plt.grid('on')
# plt.grid(True, which='both')

# er=np.sqrt(np.diag(pcov))
# yerr_data=np.sqrt(y_data)
# plt.errorbar(xmodel,y_data,yerr_data, ls='', color='k')
# plt.scatter(x_data, y_data)
# plt.plot.show()