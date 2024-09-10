import tensorflow as tf
import matplotlib.pyplot as plt

# Getting data and splitting into training and testing sets
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Scaling down images from 0-255 to 0-1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Printing information
print(train_images.shape)
print(test_images.shape)
print(train_labels)

# Displaying first image
#plt.imshow(train_images[0], cmap='gray')
#plt.show()

# Defining neural net
image_model = tf.keras.models.Sequential()
image_model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
image_model.add(tf.keras.layers.Dense(128, activation='relu'))
image_model.add(tf.keras.layers.Dense(128, activation='relu'))
image_model.add(tf.keras.layers.Dense(10, activation='softmax'))

# Compiling model
image_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training model
image_model.fit(train_images, train_labels, epochs=4)

# Testing model
val_loss, val_acc = image_model.evaluate(test_images, test_labels)
print("Test accuracy: ", val_acc)
print("Test loss: ", val_loss)

# Exporting model
image_model.save('HandwrittenNumbersModel.keras')

# Loading model
image_model_new = tf.keras.models.load_model("HandwrittenNumbersModel.keras")

# Making sure it's the same model
val_loss, val_acc = image_model_new.evaluate(test_images, test_labels)
print("Test accuracy: ", val_acc)
print("Test loss: ", val_loss)