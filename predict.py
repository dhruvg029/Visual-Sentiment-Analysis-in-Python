from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

img = image.load_img('sample.jpg',target_size=(112,112))
img = np.asarray(img)
img = np.expand_dims(img, axis=0)

from keras.models import load_model

model = load_model('abc.tflite')

output = model.predict(img)[0]

for i in output:
    print(i)

