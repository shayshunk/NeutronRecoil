import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from decimal import Decimal
from tensorflow.keras.callbacks import TensorBoard  # type: ignore
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
recoils = input("Enter how many recoils to train for: ")
dataPoints = 3 * int(recoils)
path = r'/home/shashank/Documents/Projects/NeutronRecoil/Data/Continuous/' + \
    recoils + '_Recoils/'

smearing = input(
    "Do you want detector smearing? (y/n) ").lower().strip() == 'y'

if (smearing):
    files = glob.glob(os.path.join(path, "ContinuousTraining_*.pkl"))
else:
    files = glob.glob(os.path.join(path, "ContinuousTraining_NoSmear*.pkl"))

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
print(len(recoilData), "total events")

# Splitting data into training and testing labels, saving 25% for testing
trainingData = recoilData.iloc[:14000000, :]
testingData = recoilData.iloc[14000000:, :]

# Extracting proton data vs neutron data
neutronTraining = trainingData.iloc[:, dataPoints]
protonTraining = trainingData.drop(dataPoints, axis=1)

neutronTesting = testingData.iloc[:, dataPoints]
protonTesting = testingData.drop(dataPoints, axis=1)

# Grabbing last 1000 data points for plotting
plottingProtons = protonTesting.iloc[-1000:, :]
plottingNeutrons = neutronTesting.iloc[-1000:]

plottingProtons.reset_index(drop=True, inplace=True)
plottingNeutrons.reset_index(drop=True, inplace=True)
plottingNeutrons = plottingNeutrons * 5

# Defining model
denseLayers = [5]
layerSizes = [512]
batchSizes = [200]

counter = 1

for denseLayer in denseLayers:
    for layerSize in layerSizes:
        for batchSize in batchSizes:

            if counter < 0:
                counter = counter + 1
                continue

            NAME = "Continuous-{}-dense-{}-nodes-{}-batch-{}".format(
                denseLayer, layerSize, batchSize, int(time.time()))
            if (smearing == False):
                NAME = NAME + "-NoSmear"
            print("Model:")
            print(NAME)
            tensorboard = TensorBoard(
                log_dir='logs/Continuous/{}'.format(NAME))

            input_layer = tf.keras.layers.Input(shape=(dataPoints,))
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
                protonTesting, neutronTesting), epochs=10, callbacks=[tensorboard])

            # Saving model
            path = r'/home/shashank/Documents/Projects/NeutronRecoil/Networks/Continuous/' + \
                recoils + '_Recoils/'
            RecoilModel.save(path + "{}.keras".format(NAME))

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

            path = r'/home/shashank/Documents/Projects/NeutronRecoil/Plots/Continuous/' + \
                recoils + '_Recoils/'
            plt.savefig(path + "True Energy Continuous_{}.png".format(
                NAME), bbox_inches='tight')

            counter = counter+1
