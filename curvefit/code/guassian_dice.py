# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 17:20:50 2023

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# define the probabilities of each value on the super dice
probabilities = [1/15,1/15,1/15,1/15,1/15,2/15,4/15,2/15,1/15,1/15]

# generate a random sample of 1000 rolls using the defined probabilities
sample = np.random.choice(range(1, 11), size=1000, p=probabilities)

# define the Gaussian function
def gaussian(x, A, mu, sigma):
    return A*np.exp(-(x-mu)**2/(2*sigma**2))

 # A*np.exp(-(x-mu)**2/(2*sigma**2))
# 1/(2*np.pi*sigma)*np.exp( -(x-mu)**2 / (2*sigma2) )

# def gaussian(x, mu, sigma):
#     return 1/(np.sqrt(2*np.pi)*sigma)*np.exp( -(x-mu)**2 / (2*sigma) )

# bin the sample data into a histogram
hist, bins = np.histogram(sample, bins=range(1, 12), density=True)

# calculate the center of each bin as x values for fitting
x = (bins[:-1] + bins[1:]) / 2

x_data = x[3:10]

# fit the Gaussian model to the histogram data
popt, cov = curve_fit(gaussian, x, hist)



A, mu, sigma = popt
# plot the histogram and fitted Gaussian curve
x_model = np.linspace(4.5, 10.5, 10)
y_model = gaussian(x, A, mu, sigma)
r=hist-y_model
plt.hist(sample, bins=range(1, 12), density=True, alpha=0.5, label='Sample data')
plt.plot(x, y_model, color='red', label='Fit')
plt.title('Gaussian Curve fitting')
plt.grid('on')
plt.xlabel('Dice Number')
plt.ylabel('Probability')
plt.legend()
plt.show()
plt.plot(x,r)
plt.scatter(x,r)
plt.xlabel('Dice Number')
plt.ylabel('Difference')
plt.title('Residual gaussian fit')
plt.grid('on')

# # er=np.sqrt(np.diag(pcov))
# yerr_data=np.sqrt(hist)
# plt.errorbar(x,hist,yerr_data, ls='', color='k')
# # plt.scatter(x_data, y_data)
# plt.plot.show()