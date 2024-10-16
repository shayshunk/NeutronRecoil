import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from decimal import Decimal

plt.rcParams.update({'font.size': 16})

# Naming columns since CSVs have no header
columns = ["Phi", "Theta", "Energy"]

# Reading in truth and reconstructed data
truthData = pd.read_csv("TruthData.csv", header=None, names=columns)
realData = pd.read_csv("SmearData.csv", header=None, names=columns)

# Splitting data into training and testing labels, saving 25% for testing
trainingRecoils = truthData.iloc[:750000, :]
testingRecoils = truthData.iloc[750000:, :]

trainingSmears = realData.iloc[:750000, :]
testingSmears = realData.iloc[750000:, :]

# Defining model
RecoilModel = tf.keras.models.Sequential()
RecoilModel.add(tf.keras.layers.Dense(128, input_shape=(3,), activation="relu"))
RecoilModel.add(tf.keras.layers.Dense(128, activation="relu"))
RecoilModel.add(tf.keras.layers.Dense(128, activation="relu"))
RecoilModel.add(tf.keras.layers.Dense(3))

# Compiling model
RecoilModel.compile(
    optimizer="adam", loss="mean_squared_error", metrics=["accuracy"]
)

# Training model
RecoilModel.fit(trainingSmears, trainingRecoils, epochs=3)

# Testing model
valLoss, valAcc = RecoilModel.evaluate(testingSmears, testingRecoils)
print("Test accuracy: ", valAcc)
print("Test loss: ", valLoss)

# Grabbing last 200 data points for plotting
plottingRecoils = truthData.iloc[-200:, :]
plottingSmears = realData.iloc[-200:, :]

# Predicting Phi
modelPrediction = RecoilModel.predict(plottingSmears)

phiDiff = plottingRecoils["Phi"] - modelPrediction[:, 0]
thetaDiff = plottingRecoils["Theta"] - modelPrediction[:, 1]
energyDiff = plottingRecoils["Energy"] - modelPrediction[:, 2]

phiDiff = sum(phiDiff)/len(phiDiff)
thetaDiff = sum(thetaDiff)/len(thetaDiff)
energyDiff = sum(energyDiff)/len(energyDiff)

phiOriginalDiff = plottingSmears["Phi"] - plottingRecoils["Phi"]
phiOriginalDiff = sum(phiOriginalDiff)/len(phiOriginalDiff)
print(phiOriginalDiff)

# Plotting
xrange = np.linspace(-1.6, 1.6, 200)

plt.figure(1)
plt.plot(plottingRecoils["Phi"], modelPrediction[:,0], linestyle='None', marker='o', label="Unsmeared")
plt.plot(plottingSmears["Phi"], modelPrediction[:,0], linestyle='None', marker='o', label="Original")
plt.plot(xrange, xrange)
plt.text(-1.5, 1, "Average difference: %.3f"%(Decimal(phiDiff)), bbox = dict(facecolor = 'red', alpha = 0.5))
plt.title("Predicted Phi Recoil vs True Phi")
plt.xlabel("True Phi (radians)")
plt.xlim(-1.8, 1.8)
plt.ylabel("Predicted Phi (radians)")
plt.ylim(-1.8, 1.8)
plt.savefig("Phi.pdf", bbox_inches='tight')
plt.legend()

plt.figure(2)

xrange = np.linspace(0.2, 3, 200)

plt.plot(plottingRecoils["Theta"], modelPrediction[:,1], linestyle='None', marker='o')
plt.plot(xrange, xrange)
plt.text(0.1, 2.8, "Average difference: %.3f"%(Decimal(thetaDiff)), bbox = dict(facecolor = 'red', alpha = 0.5))
plt.title("Predicted Theta Recoil vs True Theta")
plt.xlabel("True Theta (radians)")
plt.xlim(0, 3.2)
plt.ylabel("Predicted Theta (radians)")
plt.ylim(0, 3.2)
plt.savefig("Theta.pdf", bbox_inches='tight')

plt.figure(3)

xrange = np.linspace(0, 1, 200)

plt.plot(plottingRecoils["Energy"], modelPrediction[:,2], linestyle='None', marker='o')
plt.plot(xrange, xrange)
plt.title("Predicted Energy Recoil vs True Energy")
plt.text(0, 1, "Average difference: %.3f"%(Decimal(energyDiff)), bbox = dict(facecolor = 'red', alpha = 0.5))
plt.xlabel("True Energy (MeV)")
plt.xlim(-0.2, 1.2)
plt.ylabel("Predicted Energy (MeV)")
plt.ylim(-0.2, 1.2)
plt.savefig("Energy.pdf", bbox_inches='tight')

plt.show()

print(Decimal(phiDiff))