import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# Load the MNIST dataset with training and testing splits
(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

# Get one image and its label from the testing dataset
image, label = next(iter(ds_test))

# Plot the image using matplotlib
plt.imshow(image.numpy()[:,:,0], cmap='gray')
plt.title('Label: {}'.format(label.numpy()))
plt.axis('off')
plt.show()