import tensorflow as tf
from skimage import data
import skimage
import numpy as np
import matplotlib.pyplot as plt
import os

# # Initialize two constants
# x1 = tf.constant([1, 2, 3, 4])
# x2 = tf.constant([5, 6, 7, 8])

# result = tf.multiply(x1, x2)
# # Initialize a session and do the calculation
# with tf.Session() as sess:
# 	output = sess.run(result)
# 	print(output)

def load_data(data_directory):
  directories = [d for d in os.listdir(data_directory) 
                 if os.path.isdir(os.path.join(data_directory, d))]
  labels = []
  images = []
  for d in directories:
    label_directory = os.path.join(data_directory, d)
    file_names = [os.path.join(label_directory, f) 
                  for f in os.listdir(label_directory) 
                  if f.endswith(".ppm")]
    for f in file_names:
      images.append(skimage.data.imread(f))
      labels.append(int(d))
  return images, labels

ROOT_PATH = "/Users/tomdickson/Documents/rand_dev/tensorflow"
train_data_directory = os.path.join(ROOT_PATH, "TrafficSigns/Training")
test_data_directory = os.path.join(ROOT_PATH, "TrafficSigns/Testing")

images, labels = load_data(train_data_directory)

images = np.array(images)
labels = np.array(labels)

# # Print the `images` dimensions
# print(images.ndim)
# # Print the number of `images`'s elements
# print(images.size)
# # Print the first instance of `images`
# print(images[0])

# # Print the `labels` dimensions
# print(labels.ndim)
# # Print the number of `labels`'s elements
# print(labels.size)
# # Count the number of labels
# print(len(set(labels)))

# Import the `pyplot` module of `matplotlib`

# # Determine the (random) indexes of the images that you want to see 
# traffic_signs = [300, 2250, 3650, 4000]

# # Fill out the subplots with the random images that you defined 
# for i in range(len(traffic_signs)):
#   plt.subplot(1, 4, i+1)
#   plt.axis('off')
#   plt.imshow(images[traffic_signs[i]])
#   plt.subplots_adjust(wspace=0.5)

# plt.show()

from skimage import transform 

# Rescale the images in the `images` array so that all images get the same size
images28 = [transform.resize(image, (28, 28)) for image in images]

# Import `rgb2gray` from `skimage.color`
from skimage.color import rgb2gray

# Convert `images28` to an array
images28 = np.array(images28)

# Convert `images28` to grayscale
images28 = rgb2gray(images28)

" Create the model "
# Initialize placeholders 
x = tf.placeholder(dtype = tf.float32, shape = [None, 28, 28])
y = tf.placeholder(dtype = tf.int32, shape = [None])

# Flatten the input data
images_flat = tf.contrib.layers.flatten(x)

# Fully connected layer 
logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)

# Define a loss function
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, 
                                                                    logits = logits))
# Define an optimizer 
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

# Convert logits to label indexes
correct_pred = tf.argmax(logits, 1)

# Define an accuracy metric
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

" Run the model "
tf.set_random_seed(1234)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(201):
        print('EPOCH', i)
        _, accuracy_val = sess.run([train_op, accuracy], feed_dict={x: images28, y: labels})
        if i % 10 == 0:
            print("Loss: ", loss)
        print('DONE WITH EPOCH')

# Load the test data
test_images, test_labels = load_data(test_data_directory)

# Transform the images to 28 by 28 pixels
test_images28 = [transform.resize(image, (28, 28)) for image in test_images]

# Convert to grayscale
from skimage.color import rgb2gray
test_images28 = rgb2gray(np.array(test_images28))

# Run predictions against the full test set.
predicted = sess.run([correct_pred], feed_dict={x: test_images28})[0]

# Calculate correct matches 
match_count = sum([int(y == y_) for y, y_ in zip(test_labels, predicted)])

# Calculate the accuracy
accuracy = match_count / len(test_labels)

# Print the accuracy
print("Accuracy: {:.3f}".format(accuracy))

sess.close()