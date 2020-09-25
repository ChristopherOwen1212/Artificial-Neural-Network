import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

# Apply PCA to get 3 highest principal components
def load_dataset():
    raw_data = pd.read_csv('O202-COMP7117-BS01-00-clustering.csv')
    gender = {
        'Male': 0, 
        'Female': 1
    } 
    raw_data['gender'] = [gender[x] for x in raw_data['gender']]
    return raw_data

def get_mean(data):
    return np.mean(data, axis=0)

def pca(dataset):
    pca = PCA(n_components=3)
    pca = pca.fit(dataset)
    dataset = pca.transform(dataset)

    return dataset

dataset = load_dataset()
mean = get_mean(dataset)
subtracted = dataset - mean
dataset = pca(dataset)

# Self-Organizing Map
epoch = 5000
width = 3
height = 3
input_dimension = 3

class SOM:
    def __init__(self, width, height, input_dimension):
        self.width = width
        self.height = height
        self.input_dimension = input_dimension

        self.number_node = width * height
        self.weight = tf.Variable(tf.random_normal([self.number_node, self.input_dimension]))
        self.input = tf.placeholder(tf.float32, [self.input_dimension])
        self.location = [tf.to_float([y,x]) for y in range(height) for x in range(width)]

        # Find best matching unit
        bmu = self.get_bmu()

        # Update its neighbor's weights
        self.update_weight = self.update_neighbor(bmu)

    def get_bmu(self):
        distance = self.get_distance(self.input, self.weight)
        bmu_index = tf.argmin(distance)        

        bmu_location = tf.to_float([tf.div(bmu_index, width), tf.mod(bmu_index, width)])

        return bmu_location

    def get_distance(self, node_a, node_b):
        squared_difference = tf.square(node_a - node_b)
        total_squared_difference = tf.reduce_sum(squared_difference, axis=1)

        distance = tf.sqrt(total_squared_difference)

        return distance


    def update_neighbor(self, bmu):
        sigma = tf.to_float(tf.maximum(self.height, self.width) / 2)
        distance = self.get_distance(self.location, bmu)

        # Neighbor Strength
        ns = tf.exp(tf.div(tf.negative(tf.square(distance)) , 2 * tf.square(sigma)))
        
        learning_rate = 0.1
        curr_learning_rate = ns * learning_rate

        stacked_learning_rate = tf.stack([tf.tile(tf.slice(curr_learning_rate, [i], [1]), [self.input_dimension]) for i in range(self.number_node)])

        # Calculate input and weight diff
        xw_diff = tf.subtract(self.input, self.weight)

        # Delta weight
        delta_weight = tf.multiply(stacked_learning_rate, xw_diff)

        # Update weight
        new_weight = tf.add(self.weight, delta_weight)

        return tf.assign(self.weight, new_weight)


    def train(self, dataset, epoch):
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)

            for i in range(epoch):
                for data in dataset:
                    sess.run(self.update_weight, feed_dict={
                        self.input : data
                    })

            cluster = [[] for i in range(self.height)]

            location = sess.run(self.location)
            weight = sess.run(self.weight)

            for i, loc in enumerate(location):
                cluster[ int(loc[0]) ] = weight[i]
            
            self.cluster = cluster

# Apply SOM
som = SOM(width, height, input_dimension)

# Training Data
som.train(dataset, epoch)

# Show Output
plt.imshow(som.cluster)
plt.show()
