import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder,OrdinalEncoder,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

class SOM:
    def __init__(self,height,width,input_dimension):
        self.height=height
        self.width=width
        self.input_dimension=input_dimension
        self.location = [tf.to_float([y,x]) for y in range(height) for x in range(width)]

        #row = cluster ammount
        #column = input dimension
        self.weight = tf.Variable(tf.random_normal([width*height,input_dimension]))
        self.input = tf.placeholder(tf.float32,[input_dimension])

        #get location with closest distance
        best_matching_unit = self.get_bmu()

        #update weight
        self.updated_weight,self.rate_stacked = self.update_neighbour(best_matching_unit)

    def get_bmu(self):
        square_difference = tf.square(self.input-self.weight)
        distance = tf.sqrt(tf.reduce_mean(square_difference,axis = 1))

        bmu_index = tf.argmin(distance)
        bmu_location = tf.to_float([tf.div(bmu_index,self.width),tf.mod(bmu_index,self.width)])

        return bmu_location

    def update_neighbour(self, bmu):
        learning_rate = 0.1
        sigma = tf.to_float(tf.maximum(self.width, self.height)/ 2)
        # Calculate distance for each cluster from winning node 
        square_difference = tf.square(self.location - bmu) 
        distance = tf.sqrt(tf.reduce_mean(square_difference, axis=1)) 
        neighbour_strength = tf.exp(tf.div(tf.negative(tf.square(distance)), 2 * tf.square(sigma))) 
        rate = neighbour_strength * learning_rate

        total_node = self.width*self.height
        rate_stacked = tf.stack([tf.tile(tf.slice(rate,[i],[1]),[self.input_dimension]) for i in range(total_node)]) 
        
        input_weight_difference = tf.subtract(self.input,self.weight)
        weight_difference = tf.multiply(rate_stacked,input_weight_difference)
        weight_new = tf.add(self.weight, weight_difference) 
        
        return tf.assign(self.weight, weight_new),rate_stacked
          
    def train(self, dataset, num_of_epoch): 
        
        init = tf.global_variables_initializer() 
        
        with tf.Session() as sess:
            sess.run(init) 
            
            for i in range(num_of_epoch):
                for data in dataset:
                    sess.run(self.updated_weight, feed_dict={ self.input: data }) 
            cluster = [[] for i in range(self.height)] 
            location = sess.run(self.location) 
            weight = sess.run(self. weight) 
                    
            for i, loc in enumerate(location): 
                print(i, loc[0]) 
                cluster[int(loc[0])].append(weight[i]) 
                    
            self.cluster = cluster


def main():
    df = pd.read_csv("som.csv")

    features = df[["SpecialDay","VisitorType","Weekend","ProductRelated_Duration","ExitRates"]]
    labelencoder = LabelEncoder()

    features[["SpecialDay"]]=np.where(features["SpecialDay"].str.contains("HIGH"), 2,np.where(features["SpecialDay"].str.contains("NORMAL"), 1, 0 ))
    features[["VisitorType"]]=np.where(features["VisitorType"].str.contains("Returning_Visitor"), 2,np.where(features["VisitorType"].str.contains("New_Visitor"), 1, 0 ))
    features[["Weekend"]]=labelencoder.fit_transform(features['Weekend'])

    pca = PCA(n_components=5)
    principalComponents = pca.fit_transform(features)
    principalDf = pd.DataFrame(data = principalComponents, columns = ['PCA1', 'PCA2','PCA3','PCA4','PCA5'])
    principalDf = principalDf[["PCA1","PCA4","PCA5"]]
    principalDf=principalDf.values
    height=3
    width=3

    input_dimension = 3

    som=SOM(height,width,input_dimension)


    som.train(principalDf,5000)
    plt.imshow(som.cluster)
    plt.show()


main()
    


