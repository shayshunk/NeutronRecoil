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
trainingData = recoilData.iloc[:2400000, :]
testingData = recoilData.iloc[2400000:, :]

# Extracting proton data vs neutron data
neutronTraining = trainingData.iloc[:, 60]
protonTraining = trainingData.drop(60, axis=1)

print(protonTraining.head(10))
print(neutronTraining.head(10))
print(recoilData.head(10))

neutronTesting = testingData.iloc[:, 60]
protonTesting = testingData.drop(60, axis=1)

# Defining model
RecoilModel = tf.keras.models.Sequential()
RecoilModel.add(tf.keras.layers.Dense(256, input_shape=(60,), activation="relu"))
RecoilModel.add(tf.keras.layers.Dense(512, activation="relu"))
RecoilModel.add(tf.keras.layers.Dense(512, activation="relu"))
RecoilModel.add(tf.keras.layers.Dense(512, activation="relu"))
RecoilModel.add(tf.keras.layers.Dense(256, activation="relu"))
RecoilModel.add(tf.keras.layers.Dense(1,))

# Compiling model
RecoilModel.compile(
    optimizer="adam", loss="mean_squared_error", metrics=["mean_absolute_error", "mean_squared_error"]
)

# Training model
RecoilModel.fit(protonTraining, neutronTraining, validation_data=(protonTesting, neutronTesting), epochs=10)

# Grabbing last 200 data points for plotting
plottingProtons = protonTesting.iloc[-1000:, :]
plottingNeutrons = neutronTesting.iloc[-1000:] 

plottingProtons.reset_index(drop=True, inplace=True)
plottingNeutrons.reset_index(drop=True, inplace=True)

# Predicting Phi
modelPrediction = RecoilModel.predict(plottingProtons)

modelPrediction = (modelPrediction.ravel()) * 5
plottingNeutrons = plottingNeutrons * 5

predictionData = pd.DataFrame({"Predicted":modelPrediction, "True":plottingNeutrons})

print(predictionData.head(20))

print(plottingProtons.iloc[9])
print(plottingNeutrons.iloc[9])

# Plotting
plt.figure(1)
plt.plot(predictionData["True"], predictionData["Predicted"], linestyle='None', marker='.', markersize=4)
plt.title("Predicted Neutron Energy vs True Neutron Energy")
plt.xlabel("True Energy (MeV)")
plt.xlim(0, 5.5)
plt.ylabel("Predicted Energy (MeV)")
plt.ylim(0, 5.5)
plt.savefig("True Energy.pdf", bbox_inches='tight')