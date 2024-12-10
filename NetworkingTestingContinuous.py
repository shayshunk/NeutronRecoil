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


recoils = input("Enter number of recoils to test: ")
columns = int(recoils) * 3

smearing = input(
    "Testing with detector smearing? (y/n) ").lower().strip() == 'y'

# Reading in models
path = r'/home/shashank/Documents/Projects/NeutronRecoil/Networks/Continuous/' + \
    recoils + '_Recoils/'

if (smearing):
    fileName = "Continuous-5-dense-512-nodes-200-batch.keras"
else:
    fileName = "Continuous-5-dense-512-nodes-200-batch-NoSmear.keras"

recoilModel = tf.keras.models.load_model(path + fileName)

# Reading in truth and reconstructed data
path = r'/home/shashank/Documents/Projects/NeutronRecoil/Data/Discrete/' + \
    recoils + '_Recoils/'

if (smearing):
    fileName = "DiscreteTesting_*.pkl"
else:
    fileName = "DiscreteTesting-NoSmear_*.pkl"

datasets = glob.glob(os.path.join(path, fileName))

recoilDatasets = []
testingEnergies = []
bin = 100

for dataset in datasets:
    print("Reading in dataset: ", dataset)

    recoilData = pd.read_pickle(dataset)

    neutronData = recoilData.iloc[:, columns]
    neutronData = neutronData * 5

    testingEnergies.append(neutronData.iloc[0])

    recoilData = recoilData.drop(columns, axis=1)

    recoilDatasets.append(recoilData)

    del recoilData

networkBias = []
networkBiasFractional = []
networkSigma = []
networkSigmaFractional = []

path = r'/home/shashank/Documents/Projects/NeutronRecoil/Plots/Continuous/' + \
    recoils + '_Recoils/'

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
    plt.xlabel("Predicted Energy")
    plt.ylabel("Counts")

    # Add the fit parameters to the plot
    plt.figtext(0.72, 0.75, fr'$\mu$ = {popt[1]:.3f}'+'\n' +
                fr'$\sigma$ = {popt[2]:.3f}', fontsize=14)

    plt.figtext(0.15, 0.75, f'True = {energy:.2f} MeV', fontsize=14)

    if (smearing):
        fileName = f"Testing_{energy}.png"
    else:
        fileName = f"Testing_{energy}-NoSmear.png"

    plt.savefig(path + fileName, bbox_inches='tight')

    plt.close()

    mean = popt[1]
    std = popt[2]

    networkBias.append(mean - energy)
    networkSigma.append(std)
    networkBiasFractional.append((mean - energy) / energy)
    networkSigmaFractional.append(std / energy)


plt.figure(3)
fig, ax = plt.subplots()

plt.scatter(testingEnergies, networkBias)
plt.title("Bias vs Energy")
plt.xlabel("True Energy (MeV)")
plt.ylabel("Network Bias (MeV)")

if (smearing):
    fileName = "Network Biases.png"
    print(smearing)
else:
    fileName = "Network Biases-NoSmear.png"

plt.savefig(path + fileName, bbox_inches='tight')

plt.figure(4)
fig, ax = plt.subplots()

plt.scatter(testingEnergies, networkSigma)
plt.title("Uncertainty vs Energy")
plt.xlabel("True Energy (MeV)")
plt.ylabel("Network Uncertainty (MeV)")

if (smearing):
    fileName = "Network Sigmas.png"
else:
    fileName = "Network Sigmas-NoSmear.png"
plt.savefig(path + fileName, bbox_inches='tight')

plt.figure(5)
fig, ax = plt.subplots()

plt.scatter(testingEnergies, networkBiasFractional)
plt.ylim(-0.05, 0.05)
plt.title("Bias vs Energy")
plt.xlabel("True Energy (MeV)")
plt.ylabel("Network Bias / Energy")

if (smearing):
    fileName = "Network Biases Fractional.png"
    print(smearing)
else:
    fileName = "Network Biases Fractional-NoSmear.png"

plt.savefig(path + fileName, bbox_inches='tight')

plt.figure(6)
fig, ax = plt.subplots()

plt.scatter(testingEnergies, networkSigmaFractional)
plt.ylim(-0.05, 0.05)
plt.title("Uncertainty vs Energy")
plt.xlabel("True Energy (MeV)")
plt.ylabel("Network Uncertainty / Energy")

if (smearing):
    fileName = "Network Sigmas Fractional.png"
else:
    fileName = "Network Sigmas Fractional-NoSmear.png"
plt.savefig(path + fileName, bbox_inches='tight')
