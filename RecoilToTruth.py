import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from decimal import Decimal

plt.rcParams.update({'font.size': 16})

# Naming columns since CSVs have no header
columns = ["Phi", "Theta", "Proton Energy", "Neutron Energy"]

# Reading in truth and reconstructed data
recoilData = pd.read_csv("OutputData.csv", header=None, names=columns)

# Scrambling database
recoilData = recoilData.sample(frac=1).reset_index(drop=True)

# Splitting data into training and testing labels, saving 25% for testing
trainingData = recoilData.iloc[:800000, :]
testingData = recoilData.iloc[800000:, :]

# Extracting proton data vs neutron data
neutronTraining = trainingData["Neutron Energy"]
protonTraining = trainingData[["Phi", "Theta", "Proton Energy"]]

neutronTesting = testingData["Neutron Energy"]
protonTesting = testingData[["Phi", "Theta", "Proton Energy"]]

# Defining model
RecoilModel = tf.keras.models.Sequential()
RecoilModel.add(tf.keras.layers.Dense(128, input_shape=(3,), activation="relu"))
RecoilModel.add(tf.keras.layers.Dense(128, activation="relu"))
RecoilModel.add(tf.keras.layers.Dense(128, activation="relu"))
RecoilModel.add(tf.keras.layers.Dense(1))

# Compiling model
RecoilModel.compile(
    optimizer="adam", loss="mean_squared_error", metrics=["accuracy"]
)

# Training model
RecoilModel.fit(protonTraining, neutronTraining, epochs=5)

# Testing model
valLoss, valAcc = RecoilModel.evaluate(protonTesting, neutronTesting)
print("Test accuracy: ", valAcc)
print("Test loss: ", valLoss)

# Grabbing last 200 data points for plotting
plottingProtons = protonTesting.iloc[-200:, :]
plottingNeutrons = neutronTesting.iloc[-200:]

# Predicting Phi
modelPrediction = RecoilModel.predict(plottingProtons)

# Plotting
xrange = np.linspace(0, 2.0, 200)

plt.figure(1)
plt.plot(plottingNeutrons, modelPrediction, linestyle='None', marker='o', label="Unsmeared")
plt.plot(xrange, xrange)
plt.title("Predicted Neutron Energy vs True Neutron Energy")
plt.xlabel("True Energy (keV)")
plt.xlim(0, 2.1)
plt.ylabel("Predicted Energy (keV)")
plt.ylim(0, 2.1)
plt.savefig("True Energy.pdf", bbox_inches='tight')
plt.legend()