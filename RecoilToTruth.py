import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

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
RecoilModel.fit(trainingSmears, trainingRecoils, epochs=5)

# Testing model
valLoss, valAcc = RecoilModel.evaluate(testingSmears, testingRecoils)
print("Test accuracy: ", valAcc)
print("Test loss: ", valLoss)

# Grabbing last 200 data points for plotting
plottingRecoils = truthData.iloc[-200:, :]
plottingSmears = realData.iloc[-200:, :]

# Predicting Phi
modelPrediction = RecoilModel.predict(plottingSmears)

# Plotting
xrange = np.linspace(-1.6, 1.6, 200)

plt.plot(plottingRecoils["Phi"], modelPrediction[:,0], linestyle='None', marker='o')
plt.plot(xrange, xrange)
plt.title("Predicted Phi Recoil vs True Phi")
plt.xlabel("True Phi")
plt.xlim(-1.8, 1.8)
plt.ylabel("Predicted Phi")
plt.ylim(-1.8, 1.8)
plt.show()