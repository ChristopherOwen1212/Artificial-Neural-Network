import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

learning_rate = 0.01
epoch = 600
batch_size = 50
image_height = 28
image_width = 28

final_height = 7
final_width = 7

kernel_width = 3
kernel_height = 3

channel_input_1 = 1
channel_output_1 = 4
channel_input_2 = 4     # Same as previous Channel
channel_output_2 = 8    # Final Channel 

fc_input = final_height * final_width * channel_output_2
fc_output = 10          # From 0 to 9

def print_samples():
    sample_data = mnist.train.next_batch(1)
    sample_image = sample_data[0][0]
    sample_label = sample_data[1][0]

    print(sample_image)
    print(sample_label)

    sample_image = np.reshape(sample_image, [28,28])
    print(np.argmax(sample_label))
    plt.imshow(sample_image)
    plt.show()

def get_variable(shape):
    return tf.Variable(tf.random_normal(shape))

def conv2D(input, kernel):
    return tf.nn.conv2d(input, kernel, strides=[1,1,1,1], padding='SAME')

def get_conv_layer(input, w_shape, b_shape):
    w = get_variable(w_shape)
    b = get_variable(b_shape)

    y = conv2D(input, w)
    return tf.nn.relu(y+b)

def max_pool(input):
    return tf.nn.max_pool(input, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def construct_conv_layers(input):
    conv_1 = get_conv_layer(input, [kernel_height, kernel_width, channel_input_1, channel_output_1], [channel_output_1])
    max_pool_1 = max_pool(conv_1)

    conv_2 = get_conv_layer(max_pool_1, [kernel_height, kernel_width, channel_input_2, channel_output_2], [channel_output_2])
    max_pool_2 = max_pool(conv_2)

    return max_pool_2

def construct_fc_layers(input):
    w = get_variable([fc_input, fc_output])
    b = get_variable([fc_output])

    return tf.matmul(input,w) + b

# Prep Data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

x_input = tf.placeholder(tf.float32, [None, image_width*image_height])
y_target = tf.placeholder(tf.float32, [None, fc_output])

input_image = tf.reshape(x_input, [-1, image_height, image_width, 1])

# Construct Convo Layers
conv_layers = construct_conv_layers(input_image)

# Contruct Fully Connected Layers
flattened_res = tf.reshape(conv_layers, [-1,fc_input])
y_prediction = construct_fc_layers(flattened_res)

# Loss, Optimizer, Train
loss = tf.nn.softmax_cross_entropy_with_logits(logits=y_prediction, labels=y_target)
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

# Train, Test
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Training
    for i in range(epoch):
        x_batch, y_batch = mnist.train.next_batch(batch_size)
        feed = {
            x_input : x_batch,
            y_target : y_batch
        }

        sess.run(train, feed_dict=feed)

        if (i+1) % 200 == 0:
            matches = tf.equal(tf.argmax(y_prediction,1), tf.argmax(y_target, 1))

            accuracy = tf.reduce_mean(tf.cast(matches, tf.float32))

            dict_feed = {
                x_input : mnist.test.images,
                y_target : mnist.test.labels
            }

            print('Iteration #', i+1,' Accuracy: ', end='')
            print(sess.run(accuracy, feed_dict=dict_feed)*100, '%')

    # Testing
    test_data = mnist.test.next_batch(1)
    test_image = test_data[0][0]
    test_label = test_data[1][0]

    translated_label = np.argmax(test_label)
    print('Here is an image of number: ', translated_label)

    reshaped_image = np.reshape(test_image, [image_width, image_height])

    plt.imshow(reshaped_image)
    plt.show()

    print('Here is the result of predicted number: ', end='')
    feeder = {
        x_input: [test_image]
    }

    test_prediction = sess.run(y_prediction, feed_dict=feeder)
    translated_prediction = np.argmax(test_prediction)

    print(translated_prediction)
