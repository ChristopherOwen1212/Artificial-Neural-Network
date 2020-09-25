import pandas as pd 
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

def change_value_free_sulfur_dioxide(data):
    conditions = [
    (data["free sulfur dioxide"] == 'High'),
    (data["free sulfur dioxide"] == 'Medium'),
    (data["free sulfur dioxide"] == 'Low')
    ]

    value = [3, 2, 1]

    return np.select(conditions, value, default = 0)

def change_value_density(data):
    conditions = [
    (data["density"] == 'Very High'),
    (data["density"] == 'High'),
    (data["density"] == 'Medium'),
    (data["density"] == 'Low')
    ]

    value = [0, 3, 2, 1]

    return np.select(conditions, value)

def change_value_pH(data):
    conditions = [
    (data["pH"] == 'Very Basic'),
    (data["pH"] == 'Normal'),
    (data["pH"] == 'Very Acidic')
    ]

    value = [3, 2, 1]

    return np.select(conditions, value, default = 0)

def load_data():
    # Read CSV file
    data = pd.read_csv("Classification.csv")

    # Change value
    data["free sulfur dioxide"] = change_value_free_sulfur_dioxide(data)
    data["density"] = change_value_density(data)
    data["pH"] = change_value_pH(data)

    # Determine Features and Target
    features = data[["volatile acidity","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]]
    target = data[["quality"]]

    # Change Target to Numeric
    encoder = OneHotEncoder(sparse=False)
    target = encoder.fit_transform(target)

    return features,target

input_dataset,output_dataset = load_data()

def Feature_Extraction(input_dataset):
    # Normalize Dataset
    input_dataset = MinMaxScaler().fit_transform(input_dataset)
    # Extract 4 Features
    input_dataset = PCA(n_components = 4).fit_transform(input_dataset)
    return input_dataset

input_dataset = Feature_Extraction(input_dataset)

layers = {
    "input" : 4,
    "hidden" : 10,
    "output" : 5
}

weights = {
    "input_to_hidden" : tf.Variable(tf.random_normal([layers["input"],layers["hidden"]])),
    "hidden_to_output" : tf.Variable(tf.random_normal([layers["hidden"],layers["output"]]))
}

biases = {
    "input_to_hidden" : tf.Variable(tf.random_normal([layers["hidden"]])),
    "hidden_to_output" : tf.Variable(tf.random_normal([layers["output"]]))
}

input_placeholder = tf.placeholder(tf.float32,[None, layers["input"]])
output_placeholder = tf.placeholder(tf.float32,[None, layers["output"]])

def feed_forward(datas):
    input_to_hidden_plusbias = tf.matmul(datas,weights["input_to_hidden"]) + biases["input_to_hidden"]
    activation_function_sigmoid_input = tf.nn.sigmoid(input_to_hidden_plusbias)

    hidden_to_output_plusbias = tf.matmul(activation_function_sigmoid_input,weights["hidden_to_output"]) + biases["hidden_to_output"]
    activation_function_sigmoid_hidden = tf.nn.sigmoid(hidden_to_output_plusbias)

    return activation_function_sigmoid_hidden

output = feed_forward(input_placeholder)
epoch = 5000
alpha = 0.2

errors = tf.reduce_mean(0.5 * (output_placeholder - output) ** 2)

optimizer = tf.train.GradientDescentOptimizer(alpha)
train = optimizer.minimize (errors)

train_size = 0.70
validation_size = 0.20
test_size = 0.10

input_train, input_test_validation, output_train, output_test_validation = train_test_split(input_dataset, output_dataset, test_size = 1 - train_size)

input_validation, input_test, output_validation, output_test = train_test_split(input_test_validation, output_test_validation, test_size = test_size/(test_size + validation_size))

saver = tf.train.Saver()

with tf.Session() as sess : 
    sess.run(tf.global_variables_initializer())

    validationLoss = 0
    
    for i in range(1,epoch+1):
        train_dict = {
            input_placeholder : input_train,
            output_placeholder : output_train
        }

        validation_dict = {
            input_placeholder : input_validation,
            output_placeholder : output_validation
        }

        sess.run(train, feed_dict = train_dict)

        loss = sess.run(errors, feed_dict = train_dict)

        if i % 100 == 0 :
            print("Epoch : {}, Current Error : {}".format(i,loss))

        if i == 500 :
            validationLoss = sess.run(errors, feed_dict = validation_dict)
            saver.save(sess, "model/model.ckpt")

        if i % 500 == 0 :
            newValidationLoss = sess.run(errors, feed_dict = validation_dict)
            if newValidationLoss < validationLoss:
                validationLoss = newValidationLoss
                saver.save(sess, "model/model.ckpt")
    
    matches = tf.equal(tf.argmax(output_placeholder,axis = 1), tf.argmax(output,axis = 1))    
    accuracy = tf.reduce_mean(tf.cast(matches, tf.float32))

    feed_test = {
        input_placeholder : input_test,
        output_placeholder : output_test
    }

    print('Accuracy : {:.3f}%\n'.format(sess.run(accuracy, feed_dict = feed_test)*100))      
