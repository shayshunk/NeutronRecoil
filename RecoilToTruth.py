import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from decimal import Decimal
from tensorflow.keras.callbacks import TensorBoard
import time
import glob
import os

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(
                memory_limit=8000)]  # 9GB limit
        )
    except RuntimeError as e:
        print(e)

plt.rcParams.update({'font.size': 16})

# Reading in truth and reconstructed data
path = r'/home/shashank/Documents/Projects/NeutronRecoil/Data/Continuous'
files = glob.glob(os.path.join(path, "ContinuousTraining_5.50_MeV.pkl"))

recoilData = pd.DataFrame()
li = []

for filename in files:
    print("Reading in: ", filename)
    df = pd.read_pickle(filename)
    li.append(df)
    del df

recoilData = pd.concat(li)
del li

# Scrambling database
recoilData = recoilData.sample(frac=1).reset_index(drop=True)
print(recoilData.head(20))

# Splitting data into training and testing labels, saving 25% for testing
trainingData = recoilData.iloc[:9000000, :]
testingData = recoilData.iloc[9000000:, :]

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
denseLayers = [3, 4]
layerSizes = [256, 512]
batchSizes = [100, 200]

counter = 1

for denseLayer in denseLayers:
    for layerSize in layerSizes:
        for batchSize in batchSizes:

            if counter < 0:
                counter = counter + 1
                continue

            NAME = "Continuous-{}-dense-{}-nodes-{}-batch-{}".format(
                denseLayer, layerSize, batchSize, int(time.time()))
            print("Model:")
            print(NAME)
            tensorboard = TensorBoard(
                log_dir='logs/Continuous/{}'.format(NAME))

            input_layer = tf.keras.layers.Input(shape=(60,))
            RecoilModel = tf.keras.models.Sequential()
            RecoilModel.add(tf.keras.layers.Dense(
                layerSize, activation="relu"))

            for i in range(denseLayer - 1):
                RecoilModel.add(tf.keras.layers.Dense(
                    layerSize, activation="relu"))

            RecoilModel.add(tf.keras.layers.Dense(1,))

            # Compiling model
            RecoilModel.compile(
                optimizer="adam", loss="mean_squared_error", metrics=["mean_absolute_error"]
            )

            # Training model
            RecoilModel.fit(protonTraining, neutronTraining, batch_size=batchSize, validation_data=(
                protonTesting, neutronTesting), epochs=4, callbacks=[tensorboard])

            # Saving model
            RecoilModel.save("Networks/Continuous/{}.keras".format(NAME))

            # Predicting Phi
            modelPrediction = RecoilModel.predict(plottingProtons)

            modelPrediction = (modelPrediction.ravel()) * 5

            predictionData = pd.DataFrame(
                {"Predicted": modelPrediction, "True": plottingNeutrons})
            predictionData["Error"] = (
                (predictionData["True"] - predictionData["Predicted"]) / predictionData["True"]) * 100
            predictionData["Error"] = predictionData["Error"].abs()

            print(predictionData[predictionData["Error"] >= 20])

            """ print(plottingProtons.iloc[9])
            print(plottingNeutrons.iloc[9]) """

            # Plotting
            plt.figure(counter)
            plt.plot(predictionData["True"], predictionData["Predicted"],
                     linestyle='None', marker='.', markersize=4)

            xrange = np.linspace(0, 5.1, 200)
            plt.plot(xrange, xrange)
            plt.title("Predicted Neutron Energy vs True Neutron Energy")
            plt.xlabel("True Energy (MeV)")
            plt.xlim(0, 5.5)
            plt.ylabel("Predicted Energy (MeV)")
            plt.ylim(0, 5.5)
            plt.savefig(
                "Plots/Continuous/True Energy Continuous_{}.png".format(NAME), bbox_inches='tight')

            counter = counter+1
