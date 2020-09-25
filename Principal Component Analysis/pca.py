# Principal component analysis

from scipy.io import loadmat
# buat import file .mat
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import numpy as np
import matplotlib.pyplot as plt


# 1. Load Dataset
def load_dataset():
    dataset = loadmat("olivettifaces.mat")
    # isinya gambar muka dengan ukuran matrix 400 x 4096
    default_images = dataset['faces']
    transposed_images = np.transpose(default_images)

    return transposed_images.astype(float), default_images

dataset, original_dataset = load_dataset()

# 2. Calculate Mean
def calc_mean(dataset):
    return tf.reduce_mean(dataset, axis=0)

mean = calc_mean(dataset)

# 3. Normalize Data (Data - mean)
def normalize_data(dataset, mean):
    return dataset - mean

norm_dataset= normalize_data(dataset, mean)

# 4. Calculate Covariance
# Matrix Covariance = Matrix * Matrix T (matrix yang udah ditranspose)
def calc_covariance(dataset):
    transposed_dataset = tf.transpose(dataset)
    return tf.matmul(dataset, transposed_dataset)

covariance = calc_covariance(norm_dataset)

# 5. Calculate Eigen Vector
def calc_eigen_vector(covariance):
    eigen_value, eigen_vector = tf.self_adjoint_eig(covariance)
    return tf.reverse(eigen_vector, [1]) 
    #yang diflip cuman 1 dimensi

    # reverse buat ngeflip setiap sequence array
    # Before
    # [[1 2 3 4],
    #  [3 4 5 6]]

    # After
    # [[4 3 2 1],
    #  [6 5 4 3]]

eigen_vector = calc_eigen_vector(covariance)

# 6. Calculate Eigen Face
def calc_eigen_face(dataset, eigen_vector):
    # Pengenalan wajah berdasarkan eigen vector
    transposed_dataset = tf.transpose(dataset)
    return tf.transpose(tf.matmul(transposed_dataset, eigen_vector))

eigen_face = calc_eigen_face(norm_dataset, eigen_vector)

# 7. Prepare and Show image
def show_image(image):
    reshaped_image = image.reshape(64,64)
    # karena matrix awalnya 4096 lalu dipecah (pake akar) jadi 64 * 64
    plt.imshow(reshaped_image)
    plt.show()

with tf.Session() as sess:
    result = sess.run(eigen_face)

show_image(result[5])
# bisa dipilih image ke berapa (masukin ke array)
