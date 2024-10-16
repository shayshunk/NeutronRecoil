import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from decimal import Decimal

plt.rcParams.update({'font.size': 16})

# Reading in truth and reconstructed data
recoilData = pd.read_csv("OutputData.csv", header=None)

# Scrambling database
recoilData = recoilData.sample(frac=1).reset_index(drop=True)

print("Reading in data")
print(recoilData.head())
print("Max and min of each column")
print(recoilData.max(axis=0))
print(recoilData.min(axis=0))

# Splitting data into training and testing labels, saving 25% for testing
trainingData = recoilData.iloc[:240000, :]
testingData = recoilData.iloc[240000:, :]

# Extracting proton data vs neutron data
neutronTraining = trainingData.iloc[:, 60]
protonTraining = trainingData.drop(60, axis=1)

neutronTesting = testingData.iloc[:, 60]
protonTesting = testingData.drop(60, axis=1)

# Defining model
RecoilModel = tf.keras.models.Sequential()
RecoilModel.add(tf.keras.layers.Dense(64, input_shape=(60,), activation="relu"))
RecoilModel.add(tf.keras.layers.Dense(128, activation="relu"))
RecoilModel.add(tf.keras.layers.Dense(128, activation="relu"))
RecoilModel.add(tf.keras.layers.Dense(128, activation="relu"))
RecoilModel.add(tf.keras.layers.Dense(64, activation="relu"))
RecoilModel.add(tf.keras.layers.Dense(3, activation="softmax"))

# Compiling model
RecoilModel.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy", "mean_squared_error"]
)

# Training model
RecoilModel.fit(protonTraining, neutronTraining, validation_data=(protonTesting, neutronTesting), epochs=5)

# Testing model
valLoss, valAcc = RecoilModel.evaluate(protonTesting, neutronTesting)
print("Test accuracy: ", valAcc)
print("Test loss: ", valLoss)

# Grabbing last 200 data points for plotting
plottingProtons = protonTesting.iloc[-200:, :]
plottingNeutrons = neutronTesting.iloc[-200:] 

# Predicting Phi
modelPrediction = RecoilModel.predict(plottingProtons)

modelPrediction = modelPrediction * 5
plottingNeutrons = plottingNeutrons * 5

# Plotting
plt.figure(1)
plt.plot(plottingNeutrons, modelPrediction, linestyle='None', marker='.', markersize=12)
plt.title("Predicted Neutron Energy vs True Neutron Energy")
plt.xlabel("True Energy (MeV)")
plt.xlim(0, 5.5)
plt.ylabel("Predicted Energy (MeV)")
plt.ylim(0, 5.5)
plt.savefig("True Energy.pdf", bbox_inches='tight')
plt.legend()