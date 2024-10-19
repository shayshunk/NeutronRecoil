import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from decimal import Decimal
import scipy as sp 
from scipy.optimize import curve_fit

plt.rcParams.update({'font.size': 16})

bestModel1 = tf.keras.models.load_model("BestModel1.keras")
bestModel2 = tf.keras.models.load_model("BestModel2.keras")
worstModel1 = tf.keras.models.load_model("WorstModel1.keras")
worstModel2 = tf.keras.models.load_model("WorstModel2.keras")

# Reading in truth and reconstructed data
recoilData = pd.read_csv("TestingData.csv", header=None)

recoilData = recoilData.drop(60, axis=1)

prediction1 = bestModel1.predict(recoilData)
prediction2 = bestModel2.predict(recoilData)
prediction3 = worstModel1.predict(recoilData)
prediction4 = worstModel2.predict(recoilData)

prediction1 = (prediction1.ravel()) * 5
prediction2 = (prediction2.ravel()) * 5
prediction3 = (prediction3.ravel()) * 5
prediction4 = (prediction4.ravel()) * 5

# Define the Gaussian function
def gaussian(x, mu, sigma, amp):
    return amp * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

bins = 50

# Plotting
plt.figure(1)
hist, binning, _ = plt.hist(prediction1, bins, color='green')
# Fit the Gaussian curve to the histogram
popt, pcov = curve_fit(gaussian, binning[:-1], hist, p0=[1, 2, 1], maxfev=50000)
print(popt)

# Plot the fitted curve
x = np.linspace(0,3, 100)
#plt.plot(x, gaussian(x, *popt), color='red', linewidth=2)
plt.title("Neural Network Performance")
plt.xlabel("Predicted Energy (MeV)")
plt.ylabel("Counts")
plt.savefig("model1.png", bbox_inches='tight')

plt.figure(2)
plt.hist(prediction2, bins, color='green')
plt.title("Neural Network Performance")
plt.xlabel("Predicted Energy (MeV)")
plt.ylabel("Counts")
plt.savefig("model2.png", bbox_inches='tight')

plt.figure(3)
plt.hist(prediction3, bins, color='green')
plt.title("Neural Network Performance")
plt.xlabel("Predicted Energy (MeV)")
plt.ylabel("Counts")
plt.savefig("model3.png", bbox_inches='tight')

plt.figure(4)
plt.hist(prediction4, bins, color='green')
plt.title("Neural Network Performance")
plt.xlabel("Predicted Energy (MeV)")
plt.ylabel("Counts")
plt.savefig("model4.png", bbox_inches='tight')