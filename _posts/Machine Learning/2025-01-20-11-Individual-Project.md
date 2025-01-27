---
title: "11 Individual Project: Training a CNN model"
category: Machine Learning
---

In this post, I document my process for training and evaluating the 4th and final model shown in my presentation.


# Import Data

Fixing seeds and configuring my hardware.
```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from numpy.random import seed
seed(3141)

import tensorflow as tf
tf.random.set_seed(2718)
```


```python
from keras.datasets import cifar10

(X_train_all, y_train_all), (X_test, y_test) = cifar10.load_data()

label_nums = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
```

# Check Data
There are 50000 images available for training and validation and 10000 images for testing. Each image is 32 by 32 pixels with 3 colour channels.


```python
print(X_train_all.shape)
print(X_test.shape)
```

    (50000, 32, 32, 3)
    (10000, 32, 32, 3)



```python
import matplotlib.pyplot as plt

plt.imshow(X_train_all[3])
label_nums[y_train_all[3][0]]
```




    'deer'




    
![png](/Essex-MSc-Artificial-Intelligence-ePortfolio/assets/images/cnn_model_v4_5_1.png)
    


# Format Data
Data needs to be scaled to 0-1 and the labels need to be one-hot encoded.


```python
from tensorflow.keras.utils import to_categorical

X_train_all = X_train_all / 255.0
X_test = X_test / 255.0

y_train_cat_all = to_categorical(y_train_all, 10)
y_test_cat = to_categorical(y_test, 10)
```

# Split Data

A 90-10 split works well for training a CNN, maximizing training data is very important.

```python
from sklearn.model_selection import train_test_split

validation_size = 0.1

X_train, X_validation, y_train, y_validation = train_test_split(
    X_train_all, y_train_cat_all, test_size=validation_size, stratify=y_train_cat_all
)

print(X_train.shape)
print(X_validation.shape)
```

    (45000, 32, 32, 3)
    (5000, 32, 32, 3)


# Create Model


```python
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, RandomTranslation, Dropout

model = Sequential()

model.add(Input(shape=(32, 32, 3)))

model.add(RandomTranslation(height_factor=0.1, width_factor=0.1, seed=2718))

model.add(Conv2D(filters=16, kernel_size=(5, 5), padding="same", activation="relu"))

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(
    Conv2D(filters=16, kernel_size=(4, 4), padding="same", activation="relu")
)

model.add(MaxPool2D(pool_size=(4, 4)))

model.add(Flatten())

model.add(Dense(units=32, activation="relu"))

model.add(Dense(units=10, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.summary()
```

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "sequential"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ random_translation              │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">3</span>)      │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">RandomTranslation</span>)             │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>)     │         <span style="color: #00af00; text-decoration-color: #00af00">1,216</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>)     │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>)     │         <span style="color: #00af00; text-decoration-color: #00af00">4,112</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>)       │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten (<span style="color: #0087ff; text-decoration-color: #0087ff">Flatten</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)            │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)             │         <span style="color: #00af00; text-decoration-color: #00af00">8,224</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">10</span>)             │           <span style="color: #00af00; text-decoration-color: #00af00">330</span> │
└─────────────────────────────────┴────────────────────────┴───────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">13,882</span> (54.23 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">13,882</span> (54.23 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>




```python
from keras.callbacks import EarlyStopping
early_stop = EarlyStopping(
    monitor="val_loss", patience=8, verbose=1
)
history = model.fit(
    X_train,
    y_train,
    epochs=64,
    validation_data=(X_validation, y_validation),
    callbacks=[early_stop],
)

```

    Epoch 1/64


    2025-01-20 11:46:17.169515: W external/local_xla/xla/tsl/framework/cpu_allocator_impl.cc:83] Allocation of 552960000 exceeds 10% of free system memory.


    [1m1405/1407[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 7ms/step - accuracy: 0.2829 - loss: 1.9388

    2025-01-20 11:46:28.281394: W external/local_xla/xla/tsl/framework/cpu_allocator_impl.cc:83] Allocation of 61440000 exceeds 10% of free system memory.


    [1m1407/1407[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 7ms/step - accuracy: 0.2831 - loss: 1.9383 - val_accuracy: 0.4914 - val_loss: 1.4515
    Epoch 2/64
    [1m1407/1407[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 7ms/step - accuracy: 0.4686 - loss: 1.4818 - val_accuracy: 0.5402 - val_loss: 1.3138
    Epoch 3/64
    [1m1407/1407[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 7ms/step - accuracy: 0.5049 - loss: 1.3824 - val_accuracy: 0.5508 - val_loss: 1.2706
    Epoch 4/64
    [1m1407/1407[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 7ms/step - accuracy: 0.5298 - loss: 1.3176 - val_accuracy: 0.5830 - val_loss: 1.2013
    Epoch 5/64
    [1m1407/1407[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 7ms/step - accuracy: 0.5475 - loss: 1.2739 - val_accuracy: 0.5968 - val_loss: 1.1596
    Epoch 6/64
    [1m1407/1407[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 7ms/step - accuracy: 0.5588 - loss: 1.2372 - val_accuracy: 0.6052 - val_loss: 1.1367
    Epoch 7/64
    [1m1407/1407[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 7ms/step - accuracy: 0.5744 - loss: 1.2087 - val_accuracy: 0.6106 - val_loss: 1.1095
    Epoch 8/64
    [1m1407/1407[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 7ms/step - accuracy: 0.5828 - loss: 1.1841 - val_accuracy: 0.6166 - val_loss: 1.0909
    Epoch 9/64
    [1m1407/1407[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 7ms/step - accuracy: 0.5879 - loss: 1.1557 - val_accuracy: 0.6132 - val_loss: 1.0902
    Epoch 10/64
    [1m1407/1407[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 7ms/step - accuracy: 0.5966 - loss: 1.1439 - val_accuracy: 0.6250 - val_loss: 1.0647
    Epoch 11/64
    [1m1407/1407[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 7ms/step - accuracy: 0.6021 - loss: 1.1253 - val_accuracy: 0.6262 - val_loss: 1.0472
    Epoch 12/64
    [1m1407/1407[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 7ms/step - accuracy: 0.6127 - loss: 1.1029 - val_accuracy: 0.6200 - val_loss: 1.0686
    Epoch 13/64
    [1m1407/1407[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 7ms/step - accuracy: 0.6098 - loss: 1.0995 - val_accuracy: 0.6302 - val_loss: 1.0433
    Epoch 14/64
    [1m1407/1407[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 7ms/step - accuracy: 0.6150 - loss: 1.0925 - val_accuracy: 0.6446 - val_loss: 1.0108
    Epoch 15/64
    [1m1407/1407[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 7ms/step - accuracy: 0.6193 - loss: 1.0840 - val_accuracy: 0.6354 - val_loss: 1.0356
    Epoch 16/64
    [1m1407/1407[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 7ms/step - accuracy: 0.6248 - loss: 1.0738 - val_accuracy: 0.6412 - val_loss: 1.0163
    Epoch 17/64
    [1m1407/1407[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 7ms/step - accuracy: 0.6267 - loss: 1.0616 - val_accuracy: 0.6530 - val_loss: 1.0005
    Epoch 18/64
    [1m1407/1407[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 7ms/step - accuracy: 0.6274 - loss: 1.0540 - val_accuracy: 0.6530 - val_loss: 1.0021
    Epoch 19/64
    [1m1407/1407[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 7ms/step - accuracy: 0.6280 - loss: 1.0482 - val_accuracy: 0.6536 - val_loss: 0.9943
    Epoch 20/64
    [1m1407/1407[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 7ms/step - accuracy: 0.6318 - loss: 1.0397 - val_accuracy: 0.6508 - val_loss: 1.0016
    Epoch 21/64
    [1m1407/1407[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 7ms/step - accuracy: 0.6346 - loss: 1.0365 - val_accuracy: 0.6572 - val_loss: 0.9892
    Epoch 22/64
    [1m1407/1407[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 7ms/step - accuracy: 0.6367 - loss: 1.0218 - val_accuracy: 0.6594 - val_loss: 0.9880
    Epoch 23/64
    [1m1407/1407[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 7ms/step - accuracy: 0.6350 - loss: 1.0284 - val_accuracy: 0.6582 - val_loss: 0.9765
    Epoch 24/64
    [1m1407/1407[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 7ms/step - accuracy: 0.6415 - loss: 1.0196 - val_accuracy: 0.6562 - val_loss: 0.9878
    Epoch 25/64
    [1m1407/1407[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 7ms/step - accuracy: 0.6428 - loss: 1.0178 - val_accuracy: 0.6616 - val_loss: 0.9629
    Epoch 26/64
    [1m1407/1407[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 7ms/step - accuracy: 0.6457 - loss: 1.0124 - val_accuracy: 0.6682 - val_loss: 0.9511
    Epoch 27/64
    [1m1407/1407[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 7ms/step - accuracy: 0.6413 - loss: 1.0094 - val_accuracy: 0.6660 - val_loss: 0.9595
    Epoch 28/64
    [1m1407/1407[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 7ms/step - accuracy: 0.6488 - loss: 0.9956 - val_accuracy: 0.6696 - val_loss: 0.9527
    Epoch 29/64
    [1m1407/1407[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 7ms/step - accuracy: 0.6515 - loss: 0.9979 - val_accuracy: 0.6682 - val_loss: 0.9462
    Epoch 30/64
    [1m1407/1407[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 7ms/step - accuracy: 0.6483 - loss: 0.9914 - val_accuracy: 0.6690 - val_loss: 0.9489
    Epoch 31/64
    [1m1407/1407[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 7ms/step - accuracy: 0.6507 - loss: 0.9888 - val_accuracy: 0.6716 - val_loss: 0.9500
    Epoch 32/64
    [1m1407/1407[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 7ms/step - accuracy: 0.6553 - loss: 0.9839 - val_accuracy: 0.6756 - val_loss: 0.9366
    Epoch 33/64
    [1m1407/1407[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 7ms/step - accuracy: 0.6535 - loss: 0.9827 - val_accuracy: 0.6718 - val_loss: 0.9512
    Epoch 34/64
    [1m1407/1407[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 7ms/step - accuracy: 0.6566 - loss: 0.9786 - val_accuracy: 0.6660 - val_loss: 0.9428
    Epoch 35/64
    [1m1407/1407[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 7ms/step - accuracy: 0.6571 - loss: 0.9792 - val_accuracy: 0.6732 - val_loss: 0.9388
    Epoch 36/64
    [1m1407/1407[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 7ms/step - accuracy: 0.6567 - loss: 0.9711 - val_accuracy: 0.6700 - val_loss: 0.9416
    Epoch 37/64
    [1m1407/1407[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 7ms/step - accuracy: 0.6627 - loss: 0.9655 - val_accuracy: 0.6748 - val_loss: 0.9297
    Epoch 38/64
    [1m1407/1407[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 7ms/step - accuracy: 0.6606 - loss: 0.9642 - val_accuracy: 0.6698 - val_loss: 0.9205
    Epoch 39/64
    [1m1407/1407[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 7ms/step - accuracy: 0.6614 - loss: 0.9676 - val_accuracy: 0.6682 - val_loss: 0.9491
    Epoch 40/64
    [1m1407/1407[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 8ms/step - accuracy: 0.6620 - loss: 0.9628 - val_accuracy: 0.6764 - val_loss: 0.9214
    Epoch 41/64
    [1m1407/1407[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 7ms/step - accuracy: 0.6650 - loss: 0.9527 - val_accuracy: 0.6782 - val_loss: 0.9166
    Epoch 42/64
    [1m1407/1407[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 8ms/step - accuracy: 0.6654 - loss: 0.9493 - val_accuracy: 0.6776 - val_loss: 0.9225
    Epoch 43/64
    [1m1407/1407[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 8ms/step - accuracy: 0.6650 - loss: 0.9556 - val_accuracy: 0.6760 - val_loss: 0.9267
    Epoch 44/64
    [1m1407/1407[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 8ms/step - accuracy: 0.6656 - loss: 0.9510 - val_accuracy: 0.6732 - val_loss: 0.9464
    Epoch 45/64
    [1m1407/1407[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 7ms/step - accuracy: 0.6644 - loss: 0.9503 - val_accuracy: 0.6758 - val_loss: 0.9204
    Epoch 46/64
    [1m1407/1407[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 7ms/step - accuracy: 0.6673 - loss: 0.9423 - val_accuracy: 0.6770 - val_loss: 0.9233
    Epoch 47/64
    [1m1407/1407[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 7ms/step - accuracy: 0.6676 - loss: 0.9428 - val_accuracy: 0.6844 - val_loss: 0.9167
    Epoch 48/64
    [1m1407/1407[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 7ms/step - accuracy: 0.6686 - loss: 0.9474 - val_accuracy: 0.6818 - val_loss: 0.9181
    Epoch 49/64
    [1m1407/1407[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m10s[0m 7ms/step - accuracy: 0.6705 - loss: 0.9372 - val_accuracy: 0.6774 - val_loss: 0.9245
    Epoch 49: early stopping


# Plotting Metrics


```python
import pandas as pd
metrics = pd.DataFrame(model.history.history)
metrics.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>accuracy</th>
      <th>loss</th>
      <th>val_accuracy</th>
      <th>val_loss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>54</th>
      <td>0.671089</td>
      <td>0.926943</td>
      <td>0.6798</td>
      <td>0.915060</td>
    </tr>
    <tr>
      <th>55</th>
      <td>0.674578</td>
      <td>0.923418</td>
      <td>0.6820</td>
      <td>0.923459</td>
    </tr>
    <tr>
      <th>56</th>
      <td>0.676000</td>
      <td>0.921243</td>
      <td>0.6820</td>
      <td>0.918314</td>
    </tr>
    <tr>
      <th>57</th>
      <td>0.675444</td>
      <td>0.923534</td>
      <td>0.6790</td>
      <td>0.938752</td>
    </tr>
    <tr>
      <th>58</th>
      <td>0.675911</td>
      <td>0.922872</td>
      <td>0.6830</td>
      <td>0.915425</td>
    </tr>
  </tbody>
</table>
</div>




```python
metrics[['loss', 'val_loss']].plot()
plt.title('Training Loss Vs Validation Loss', fontsize=16)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.ylim(0, 1.8)
plt.show()
```


    
![png](/Essex-MSc-Artificial-Intelligence-ePortfolio/assets/images/cnn_model_v4_15_0.png)
    



```python
metrics[['accuracy', 'val_accuracy']].plot()
plt.title('Training Accuracy Vs Validation Accuracy', fontsize=16)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim(0, 0.8)
plt.show()
```


    
![png](/Essex-MSc-Artificial-Intelligence-ePortfolio/assets/images/cnn_model_v4_16_0.png)
    

The random translation layer has the interesting effect of boosting validation accuracy above the training accuracy. This displays a great improvement in model generalization. However, the increased amount of data available for training requires much more training time. For this models, the epoch limit was raised to 64 and patience to 8.


# Evaluate Model


```python
model.evaluate(X_test,y_test_cat)
```

    [1m 91/313[0m [32m━━━━━[0m[37m━━━━━━━━━━━━━━━[0m [1m0s[0m 2ms/step - accuracy: 0.6915 - loss: 0.9164

    2025-01-20 16:33:35.558517: W external/local_xla/xla/tsl/framework/cpu_allocator_impl.cc:83] Allocation of 122880000 exceeds 10% of free system memory.


    [1m313/313[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 2ms/step - accuracy: 0.6858 - loss: 0.9272





    [0.9381189346313477, 0.6805999875068665]




```python
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

predictions = np.argmax(model.predict(X_test), axis=-1)
```

    [1m102/313[0m [32m━━━━━━[0m[37m━━━━━━━━━━━━━━[0m [1m0s[0m 1ms/step

    2025-01-20 16:33:36.351875: W external/local_xla/xla/tsl/framework/cpu_allocator_impl.cc:83] Allocation of 122880000 exceeds 10% of free system memory.


    [1m313/313[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 2ms/step



```python
print(classification_report(y_test, predictions, target_names=label_nums))
```

                  precision    recall  f1-score   support
    
        airplane       0.73      0.71      0.72      1000
      automobile       0.75      0.84      0.80      1000
            bird       0.66      0.51      0.57      1000
             cat       0.57      0.40      0.47      1000
            deer       0.63      0.61      0.62      1000
             dog       0.65      0.49      0.56      1000
            frog       0.61      0.87      0.71      1000
           horse       0.67      0.79      0.73      1000
            ship       0.77      0.79      0.78      1000
           truck       0.74      0.79      0.76      1000
    
        accuracy                           0.68     10000
       macro avg       0.68      0.68      0.67     10000
    weighted avg       0.68      0.68      0.67     10000
    

The final model has an accuracy of 0.69, and the same for precision and recall. The f1-score is 0.68. These values are in line with the scores obtained by the training and validation sets, displaying the model’s ability to generalize to unseen data.

The precision-recall table shows that the model struggles greatly with some labels, in particular cat and bird. These results are understandable once we look at the images the network is trying to classify. 

```python
confusion_matrix(y_test, predictions)
```




    array([[707,  40,  50,  12,  15,   3,  17,  20,  93,  43],
           [ 11, 843,   5,   7,   1,   1,   8,   8,  29,  87],
           [ 69,   9, 509,  37, 107,  53, 119,  56,  17,  24],
           [ 28,  16,  57, 404,  81, 133, 166,  62,  25,  28],
           [ 28,  10,  46,  43, 608,  21, 108, 111,  17,   8],
           [  8,  10,  44, 138,  61, 493, 108, 101,  25,  12],
           [ 10,   8,  21,  29,  24,  11, 866,  15,  12,   4],
           [ 10,   6,  24,  22,  53,  32,  21, 794,   6,  32],
           [ 69,  48,  12,  12,  10,   2,   6,   5, 794,  42],
           [ 22, 129,   4,   4,   4,   4,   9,  17,  19, 788]])


If we look at the confusion matrix, we can see which labels the network gets mixed up the most. The top errors are cat and dog, automobile and truck, and horse and deer. This makes sense as these are all pairs of closely related items. Cats and dogs are medium-sized and frequently indoors. Autos and trucks are large, metal and in a city. Horses and deer are large with long legs, and typically in fields. The separation between airplane and ship is surprising considering they are both pointy with blue backgrounds.

```python
import seaborn as sns

ax = sns.heatmap(
    confusion_matrix(y_test, predictions),
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    xticklabels=label_nums,
    yticklabels=label_nums,
)
ax.set_xlabel("Prediction")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")
```




    Text(0.5, 1.0, 'Confusion Matrix')




    
![png](/Essex-MSc-Artificial-Intelligence-ePortfolio/assets/images/cnn_model_v4_22_1.png)
    


# Single prediction

In the misclassified example, it’s interesting that the network’s second choice is correct. The misclassification makes sense. Frogs are green or brown on green or brown backgrounds. I believe this demonstrates that my model is relying heavily on colour and background context, with shape being a secondary factor. This reliance on colour and background likely means that my model would do very poorly in the real world where lighting conditions and backgrounds are extremely varied.


```python
i = 65
my_image = X_test[i]
print(label_nums[y_test[i][0]])
plt.imshow(my_image)

predicted_classes = model.predict(my_image.reshape(1, 32, 32, 3))[0]
print(label_nums[np.argmax(predicted_classes)])
print(predicted_classes)


fig, ax = plt.subplots(1, 1, figsize=(9, 2), sharey=True)
ax.bar(label_nums, predicted_classes)
ax.set_title("Predicted Class Distribution")
```

    bird
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 19ms/step
    frog
    [5.2156015e-06 4.7499623e-05 1.5168312e-02 3.5644786e-03 1.4749492e-03
     1.3789933e-03 9.7712141e-01 1.2197773e-03 4.3608870e-06 1.5132356e-05]





    Text(0.5, 1.0, 'Predicted Class Distribution')




    
![png](/Essex-MSc-Artificial-Intelligence-ePortfolio/assets/images/cnn_model_v4_25_2.png)
    



    
![png](/Essex-MSc-Artificial-Intelligence-ePortfolio/assets/images/cnn_model_v4_25_3.png)
    



```python
my_image_num = 8
my_image = X_test[my_image_num]
print(label_nums[y_test[my_image_num][0]])
plt.imshow(my_image)
```

    cat

    <matplotlib.image.AxesImage at 0x764116f11330>




    
![png](/Essex-MSc-Artificial-Intelligence-ePortfolio/assets/images/cnn_model_v4_26_2.png)
    



```python
predicted_classes = model.predict(my_image.reshape(1, 32, 32, 3))[0]
actual_classes = to_categorical(y_test[my_image_num], 10)[0]
print(predicted_classes)
print(actual_classes)

fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharey=True)
axes[0].bar(label_nums, predicted_classes)
axes[0].set_title("Predicted Class Distribution")
axes[1].bar(label_nums, actual_classes)
axes[1].set_title("Actual Class Distribution")
fig.suptitle("Predicted Vs Actual Class Distribution", fontsize=16)
fig.tight_layout()

```

    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 20ms/step
    [3.8658531e-04 5.9386297e-05 3.1446539e-02 3.6979440e-01 1.3158613e-01
     3.3004549e-01 7.3866270e-02 6.2474739e-02 6.3995300e-05 2.7645109e-04]
    [0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]



    
![png](/Essex-MSc-Artificial-Intelligence-ePortfolio/assets/images/cnn_model_v4_27_1.png)
    



```python
from math import log

p = actual_classes
q = predicted_classes

cross_entropy = -sum(
    [p[i]*log(q[i]) for i in range(len(p))])
print(cross_entropy, 'nats')
```

    0.9948081073958105 nats

