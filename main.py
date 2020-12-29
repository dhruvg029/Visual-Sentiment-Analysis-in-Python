import keras,os
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

trdata = ImageDataGenerator()
traindata = trdata.flow_from_directory(directory='Train',target_size=(112,112))
tsdata = ImageDataGenerator()
testdata = tsdata.flow_from_directory(directory='Test', target_size=(112,112))

#VGG16

VGG16 = keras.applications.VGG16(input_shape=(112,112,3),include_top=False,weights='imagenet')
VGG16.trainable = False

model_vgg16 = keras.Sequential([
    VGG16,
    keras.layers.Flatten(),
    keras.layers.Dense(units=256,activation="relu"),
    keras.layers.Dense(units=256,activation="relu"),
    keras.layers.Dense(units=4, activation="softmax")
])

model_vgg16.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

#VGG19

VGG19 = keras.applications.VGG19(input_shape=(112,112,3),include_top=False,weights='imagenet')
VGG19.trainable = False

model_vgg19 = keras.Sequential([
    VGG19,
    keras.layers.Flatten(),
    keras.layers.Dense(units=256,activation="relu"),
    keras.layers.Dense(units=256,activation="relu"),
    keras.layers.Dense(units=4, activation="softmax")
])

model_vgg19.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

#ResNet101

ResNet101 = keras.applications.ResNet101(input_shape=(112,112,3),include_top=False,weights='imagenet')
ResNet101.trainable = False

model_resnet101 = keras.Sequential([
    ResNet101,
    keras.layers.Flatten(),
    keras.layers.Dense(units=256,activation="relu"),
    keras.layers.Dense(units=256,activation="relu"),
    keras.layers.Dense(units=4, activation="softmax")
])

model_resnet101.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

#ResNet152

ResNet152 = keras.applications.ResNet152(input_shape=(112,112,3),include_top=False,weights='imagenet')
ResNet152.trainable = False

model_resnet152 = keras.Sequential([
    ResNet152,
    keras.layers.Flatten(),
    keras.layers.Dense(units=256,activation="relu"),
    keras.layers.Dense(units=256,activation="relu"),
    keras.layers.Dense(units=4, activation="softmax")
])

model_resnet152.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

#InceptionV3

InceptionV3 = keras.applications.InceptionV3(input_shape=(112,112,3),include_top=False,weights='imagenet')
InceptionV3.trainable = False

model_inceptionv3 = keras.Sequential([
    InceptionV3,
    keras.layers.Flatten(),
    keras.layers.Dense(units=256,activation="relu"),
    keras.layers.Dense(units=256,activation="relu"),
    keras.layers.Dense(units=4, activation="softmax")
])

model_inceptionv3.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

print("")
print("------------------VGG16---------------------")
print("")

model_vgg16.summary()

hist_vgg16 = model_vgg16.fit_generator(steps_per_epoch=50,generator=traindata, validation_data= testdata, validation_steps=10,epochs=5)
model_vgg16.save('vgg16.h5')

print("")
print("------------------VGG19---------------------")
print("")

model_vgg19.summary()

hist_vgg19 = model_vgg19.fit_generator(steps_per_epoch=50,generator=traindata, validation_data= testdata, validation_steps=10,epochs=5)
model_vgg19.save('vgg19.h5')

print("")
print("------------------ResNet101---------------------")
print("")

model_resnet101.summary()

hist_resnet101 = model_resnet101.fit_generator(steps_per_epoch=50,generator=traindata, validation_data= testdata, validation_steps=10,epochs=5)
model_resnet101.save('resnet101.h5')

print("")
print("------------------ResNet152---------------------")
print("")

model_resnet152.summary()

hist_resnet152 = model_resnet152.fit_generator(steps_per_epoch=50,generator=traindata, validation_data= testdata, validation_steps=10,epochs=5)
model_resnet152.save('resnet152.h5')

print("")
print("------------------InceptionV3---------------------")
print("")

model_inceptionv3.summary()

hist_inceptionv3 = model_inceptionv3.fit_generator(steps_per_epoch=50,generator=traindata, validation_data= testdata, validation_steps=10,epochs=5)
model_inceptionv3.save('inceptionv3.h5')

import matplotlib.pyplot as plt

plt.plot(hist_vgg16.history['val_accuracy'])
plt.plot(hist_vgg19.history['val_accuracy'])
plt.plot(hist_resnet101.history['val_accuracy'])
plt.plot(hist_resnet152.history['val_accuracy'])
plt.plot(hist_inceptionv3.history['val_accuracy'])

plt.title('Models Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['VGG16','VGG19','ResNet101','ResNet152','InceptionV3'])

plt.show()