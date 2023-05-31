# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 14:44:19 2023

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define the parameters
g = 9.81 # acceleration due to gravity (m/s^2)
v0 = 0 # initial velocity (m/s)
h0 = 500 # initial altitude (m)
t_max = 15 # maximum time (s)
dt = 0.5 # time step (s)

# Generate the dataset
t = np.arange(0, t_max+dt, dt)
h = np.maximum(h0 - 0.5*g*t**2, 0)

# Plot the dataset as a scatter plot
plt.scatter(t, h)

# Add labels and title to the plot
plt.xlabel('Time (s)')
plt.ylabel('Altitude (m)')
plt.title('Altitude over Time')

# Show the plot
plt.show()
x_data= t[(h>0)&(h<=500)]
y_data= h[(h>0)&(h<=500)]
plt.scatter(x_data,y_data, label='Sample data')



# define the linear function
def linear(x, a, b):
    return a*x + b

# fit the linear model to the data
popt, cov = curve_fit(linear, x_data, y_data)

# print the fitted parameters
# print('Fitted parameters:')
# print('a =', params[0])
# print('b =', params[1])
a, b = popt
y_model = linear(x_data, a, b)
r = y_data - y_model
# plot the data and fitted line
# plt.plot(xdata, altitude, 'o', label='Data')
plt.plot(x_data, linear(x_data, a, b), 'r', label='Fit')
plt.xlabel('Time ($s$)')
plt.ylabel('Altitude ($m$)')
plt.title('Linear curve fitting')
plt.plot(x_data,r, color='k' ,label='Residual')
plt.grid('on')
plt.legend()
plt.show()


