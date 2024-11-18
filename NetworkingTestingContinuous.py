import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from decimal import Decimal
import scipy as sp
from scipy.optimize import curve_fit
from scipy.stats import norm
import os
import glob

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
plt.rcParams.update({'font.size': 16})
tf.get_logger().setLevel('ERROR')

# Define the Gaussian function


def fit_func(x, a, mu, sigma):
    """gaussian function used for the fit"""
    return a * sp.stats.norm.pdf(x, loc=mu, scale=sigma)


# Reading in models
path = r'/home/shashank/Documents/Projects/NeutronRecoil/Networks/Discrete/'

recoilModel = tf.keras.models.load_model(
    path+"Continuous-3-dense-512-nodes-100-batch.keras")

# Reading in truth and reconstructed data
path = r'/home/shashank/Documents/Projects/NeutronRecoil/Data/Discrete/'
datasets = glob.glob(os.path.join(path, "DiscreteTesting_*.pkl"))

recoilDatasets = []
testingEnergies = []
bin = 100

for dataset in datasets:
    print("Reading in dataset: ", dataset)

    recoilData = pd.read_pickle(dataset)

    neutronData = recoilData.iloc[:, 60]
    neutronData = neutronData * 5

    testingEnergies.append(neutronData.iloc[0])

    recoilData = recoilData.drop(60, axis=1)

    recoilDatasets.append(recoilData)

    del recoilData

networkBias = []
networkSigma = []

for energy, recoilData in zip(testingEnergies, recoilDatasets):

    model = recoilModel

    prediction = model.predict(recoilData)
    prediction = prediction * 5

    plt.figure(1)
    fig, ax = plt.subplots()

    hist, binning, _ = plt.hist(prediction, bins=bin)
    # Calculate the bin centers
    bin_centers = 0.5 * (binning[1:] + binning[:-1])

    xmin, xmax = plt.xlim()

    x = np.linspace(xmin, xmax, 250)

    # Fit the Gaussian curve to the histogram
    popt, pcov = curve_fit(
        fit_func, binning[:-1], hist, p0=[1, np.mean(prediction), np.std(prediction)], maxfev=100000)
    print("Fit parameters:", popt)

    # Plot the fitted curve
    plt.plot(binning, fit_func(binning, *popt),
             color='red', linewidth=1.5, alpha=0.85)

    plt.title("Neural Network Performance")
    plt.xlabel("Predicted Energy (MeV)")
    plt.ylabel("Counts")

    # Add the fit parameters to the plot
    plt.figtext(0.72, 0.75, fr'$\mu$ = {popt[1]:.3f}'+'\n' +
                fr'$\sigma$ = {popt[2]:.3f}', fontsize=14)

    plt.savefig(f"Plots/Discrete/Testing_{energy}.png", bbox_inches='tight')

    plt.close(1)

    mean = popt[1]
    std = popt[2]

    networkBias.append(mean - energy)
    networkSigma.append(std)


plt.figure(3)
fig, ax = plt.subplots()

plt.scatter(testingEnergies, networkBias)
plt.title("Bias vs Energy")
plt.xlabel("True Energy (MeV)")
plt.ylabel("Network Mean Prediction Bias")
plt.savefig("Network Biases.png")

plt.figure(4)
fig, ax = plt.subplots()

plt.scatter(testingEnergies, networkSigma)
plt.title("Uncertainty vs Energy")
plt.xlabel("True Energy (MeV)")
plt.ylabel("Network Uncertainty (Gaussian)")
plt.savefig("Network Sigmas.png")
