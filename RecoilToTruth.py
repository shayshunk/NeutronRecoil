import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from decimal import Decimal
from tensorflow.keras.callbacks import TensorBoard
import time

plt.rcParams.update({'font.size': 16})

# Reading in truth and reconstructed data
recoilData = pd.read_csv("OutputData.csv", header=None)

# Scrambling database
recoilData = recoilData.sample(frac=1).reset_index(drop=True)

""" print("Reading in data")
print(recoilData.head())
print("Max and min of each column")
print(recoilData.max(axis=0))
print(recoilData.min(axis=0))
 """
# Splitting data into training and testing labels, saving 25% for testing
trainingData = recoilData.iloc[:2400000, :]
testingData = recoilData.iloc[2400000:, :]

# Extracting proton data vs neutron data
neutronTraining = trainingData.iloc[:, 60]
protonTraining = trainingData.drop(60, axis=1)

neutronTesting = testingData.iloc[:, 60]
protonTesting = testingData.drop(60, axis=1)

# Grabbing last 1000 data points for plotting
plottingProtons = protonTesting.iloc[-1000:, :]
plottingNeutrons = neutronTesting.iloc[-1000:] 

plottingProtons.reset_index(drop=True, inplace=True)
plottingNeutrons.reset_index(drop=True, inplace=True)
plottingNeutrons = plottingNeutrons * 5

# Defining model
denseLayers = [2, 3, 4, 5]
layerSizes = [64, 128, 256, 512]
batchSizes = [32, 64, 128]

counter = 1

for denseLayer in denseLayers:
    for layerSize in layerSizes:
        for batchSize in batchSizes:
            NAME = "{}-dense-{}-nodes-{}-batch-{}".format(denseLayer, layerSize, batchSize, int(time.time()))
            print("Model:")
            print(NAME)
            tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

            RecoilModel = tf.keras.models.Sequential()
            RecoilModel.add(tf.keras.layers.Dense(layerSize, input_shape=(60,), activation="relu"))

            for i in range(denseLayer - 1):
                RecoilModel.add(tf.keras.layers.Dense(layerSize, activation="relu"))

            RecoilModel.add(tf.keras.layers.Dense(1,))

            # Compiling model
            RecoilModel.compile(
                optimizer="adam", loss="mean_squared_error", metrics=["mean_absolute_error"]
            )

            # Training model
            RecoilModel.fit(protonTraining, neutronTraining, batch_size=batchSize, validation_data=(protonTesting, neutronTesting), epochs=10, callbacks=[tensorboard])

            # Saving model
            RecoilModel.save("{}.keras".format(NAME))

            # Predicting Phi
            modelPrediction = RecoilModel.predict(plottingProtons)

            modelPrediction = (modelPrediction.ravel()) * 5

            predictionData = pd.DataFrame({"Predicted":modelPrediction, "True":plottingNeutrons})
            predictionData["Error"] = ((predictionData["True"] - predictionData["Predicted"]) / predictionData["True"]) * 100
            predictionData["Error"] = predictionData["Error"].abs()

            print(predictionData[predictionData["Error"] >= 20])

            """ print(plottingProtons.iloc[9])
            print(plottingNeutrons.iloc[9]) """

            # Plotting
            plt.figure(counter)
            plt.plot(predictionData["True"], predictionData["Predicted"], linestyle='None', marker='.', markersize=4)
            plt.title("Predicted Neutron Energy vs True Neutron Energy")
            plt.xlabel("True Energy (MeV)")
            plt.xlim(0, 5.5)
            plt.ylabel("Predicted Energy (MeV)")
            plt.ylim(0, 5.5)
            plt.savefig("True Energy_{}.pdf".format(NAME), bbox_inches='tight')
            
            counter = counter+1