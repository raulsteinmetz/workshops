import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

# Load the MNIST dataset with training and testing splits
(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

# Define a function to normalize the images
def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

# Preprocess the training data
ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

'''applies the normalization function normalize_img to each image in the test set in parallel
(using num_parallel_calls=tf.data.AUTOTUNE to automatically determine the number of parallel
calls based on available CPU resources).'''

# Preprocess the testing data
ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

# Define the neural network architecture
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10)
])

# Compile the model with specified loss function, optimizer, and evaluation metric
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

# Train the model
model.fit(
    ds_train,
    epochs=6,
    validation_data=ds_test,
)


# Select a random instance from the test set
sample = ds_test.take(1)
for image, label in sample:
    # Select a single image from the batch
    image = image[0]
    
    # Plot the image
    plt.imshow(np.squeeze(image.numpy()))
    plt.show()
    
    # Make a prediction with the model
    predictions = model.predict(image[np.newaxis, ...])
    predicted_label = np.argmax(predictions)
    
    print("Predicted label:", predicted_label)