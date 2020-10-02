import os 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt


# %matplotlib inline 
import warnings
warnings.filterwarnings("ignore")


import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras.models import Sequential, Model 
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.layers import GlobalMaxPool2D, Dropout, SpatialDropout2D
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, array_to_img, img_to_array
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from sklearn.model_selection import train_test_split

# warnings TensorFlow 
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

"""<hr>
<br>

##### Using Pre-trained Network
"""

# define model 
shape = Input(shape=(150, 150, 3))
model = ResNet50(include_top=True, weights="imagenet", input_tensor=shape)

# pre-process image 

path = "mountain.jpg"

img = load_img(path=path, target_size=(150, 150, 3))
img = img_to_array(img)
img = np.expand_dims(img, axis=0)
x = preprocess_input(img)

pred = model.predict(x)
print("Predicted: ", decode_predictions(pred, top=3)[0])

"""<hr>
<br>

##### Extract Features with VGG-16
"""

# VGG-16 
base_model = VGG16(include_top=False, weights="imagenet")

# Feature extractor VGG-16 
model = Model(inputs=base_model.inputs, outputs=base_model.get_layer('block4_pool').output)

path = "mountain.jpg"

img = load_img(path=path, target_size=(150, 150, 3))
img = img_to_array(img)
img = np.expand_dims(img, axis=0)
x = preprocess_input(img)

# features 
features = model.predict(x)

"""<hr>
<br>

##### Fine-Tuning
"""

images = ["mountain.jpg", "valey.jpg"]
labels = [0, 1]

new_images = []

for img in images: 
  img = load_img(img, color_mode="rgb")
  img = img_to_array(img)
# img = np.expand_dims(img, axis=0)
  img /= 255.0
  new_images.append(img)

data = pd.DataFrame({"image":new_images, "label":labels})

X = np.array(new_images)
y = np.array(data["label"])

data.head()

X.shape

X_train, _, y_train, _ = train_test_split(X,y, random_state=42)

X_train.shape

# freeze all layers 
vgg = VGG16(include_top=False, weights="imagenet", input_shape=(150,150,3))

for layer in vgg.layers:
  layer.trainable = False

# Build CNN fine-tuning 

vgg16 = vgg.output
x = Flatten()(vgg16)
x = Dense(64, activation="relu")(x)
output_layer = Dense(10, activation="softmax")(x)

model = Model(inputs=vgg.input, outputs=output_layer)
model.summary()

model.compile(optimizer=Adam(0.001), 
              loss=SparseCategoricalCrossentropy(),
              metrics=["accuracy"])

model.fit(X_train, y_train, epochs=10, batch_size=1)
