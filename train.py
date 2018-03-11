from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from utils import *
from models import MLP
import numpy as np
import pandas as pd

from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *

import matplotlib.pyplot as plt
import matplotlib.cm as cm

X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y_ = tf.placeholder(tf.float32, [None, 10])

seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Read MNIST data set (Train data from CSV file)
data = pd.read_csv('./train.csv')

# For images
images = data.iloc[:,1:].values
images = images.astype(np.float)

# For labels
labels_flat = data[[0]].values.ravel()
labels_count = np.unique(labels_flat).shape[0]

def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

labels = dense_to_one_hot(labels_flat, labels_count)
labels = labels.astype(np.uint8)

images = np.multiply(images, 1.0 / 255.0)
image_size = images.shape[1]
image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)
VALIDATION_SIZE = 2000

# Split data into training & validation
validation_images = images[:VALIDATION_SIZE]
validation_labels = labels[:VALIDATION_SIZE]

train_images = images[VALIDATION_SIZE:]
train_labels = labels[VALIDATION_SIZE:]


# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'dense', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 3000, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1_1', 100, 'Number of units in hidden layer 1 of first branch')
flags.DEFINE_integer('hidden1_2', 100, 'Number of units in hidden layer 1 of second branch')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

features = X

#print(X.get_shape().as_list())

num_supports = 1
model_func = "MLP"

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'labels_mask': tf.placeholder(tf.int32),
    'features': tf.placeholder(tf.float32, [None, 28, 28, 1]),
    'labels': tf.placeholder(tf.float32, [None, 10]),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # 11helper variable for sparse dropout
}

# Create model
with TowerContext('', is_training=True):
    model = MLP(placeholders, input_dim=784, logging=True)


#Helper functions
epochs_completed = 0
index_in_epoch = 0
num_examples = train_images.shape[0]

# serve data by batches
def next_batch(batch_size):

    global train_images
    global train_labels
    global index_in_epoch
    global epochs_completed

    start = index_in_epoch
    index_in_epoch += batch_size

    if index_in_epoch > num_examples:
        # finished epoch
        epochs_completed += 1
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train_images = train_images[perm]
        train_labels = train_labels[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return train_images[start:end], train_labels[start:end]

# visualisation variables
train_accuracies = []
validation_accuracies = []
x_range = []
# Initialize session
sess = tf.Session()

# Define model evaluation function
def evaluate(features, labels, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, labels, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)


# Init variables
with TowerContext('', is_training=True):
    sess.run(tf.global_variables_initializer())

cost_val = []

# Train model
for epoch in range(FLAGS.epochs):

    batch_X, batch_Y = next_batch(100)


    t = time.time()
    # Construct feed dictionary
    image_X = tf.placeholder(tf.float32, [None, 28, 28, 1])
    image_X = tf.Variable(batch_X)
    image_X = tf.reshape(image_X, [100,28,28,1])
    batch_X = batch_X.reshape(100,28,28,1)
    feed_dict = construct_feed_dict(batch_X, batch_Y, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)
    if(VALIDATION_SIZE):
            asd,validation_accuracy,fasd = evaluate((validation_images[0:100]).reshape(100,28,28,1),validation_labels[0:100], placeholders)
            print('validation_accuracy => %.4f'%validation_accuracy)

    # Validation
    cost, acc, duration = evaluate(batch_X, batch_Y, placeholders)
    cost_val.append(cost)

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))
    validation_accuracies.append(validation_accuracy)
    train_accuracies.append(acc)
    x_range.append(epoch+1)

print("Optimization Finished!")
plt.plot(x_range, train_accuracies,'-b', label='Training_data')
plt.plot(x_range, validation_accuracies,'-g', label='Validation_data')
plt.legend(loc='lower right', frameon=False)
plt.ylim(ymax = 1.0, ymin = 0.5)
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.show()
