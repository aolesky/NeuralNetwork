'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton
'''

import numpy as np
from numpy import ndarray
import random
import pdb
import math
class NeuralNet:

    def __init__(self, layers, epsilon=0.12, learningRate=.6, numEpochs=100):
        '''
        Constructor
        Arguments:
        	layers - a numpy array of L-2 integers (L is # layers in the network)
        	epsilon - one half the interval around zero for setting the initial weights
        	learningRate - the learning rate for backpropagation
        	numEpochs - the number of epochs to run during training
        '''
        self.layers = layers
        self.epsilon = epsilon
        self.learningRate = learningRate
        self.numEpochs = numEpochs
        self.regulization = .001

    def initializeZeroWeights(self, input_size, final_layer):
        weights = []
        for i in range(len(self.layers) + 1):
            n = 0
            d = 0
            if i == len(self.layers):
                n = final_layer
            else:
                n = self.layers[i]
            if i > 0:
                d = self.layers[i - 1]
            else:
                d = input_size

            weights.append(np.zeros((n, d + 1)))

        return weights

    def initializeWeights(self, input_size, final_layer):
        weights = []
        for i in range(len(self.layers) + 1):
            n = 0
            d = 0
            if i == len(self.layers):
                n = final_layer
            else:
                n = self.layers[i]
            if i > 0:
                d = self.layers[i - 1]
            else:
                d = input_size

            weights.append(np.empty((n, d + 1)))
            for j in range(n):
                for k in range(d + 1):
                    weights[i][j, k] = (random.random() * 2 * self.epsilon) - self.epsilon


        return weights

    def fix_y(self, y, outputs):
        result = np.zeros((y.size, outputs))
        for i in range(y.size):
            result[i, int(y[i])] = 1

        return result

    
    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy array
            y is an n-dimensional numpy array
        '''
        n, d = X.shape

        final_layer = np.unique(y).size
        adjusted_y = self.fix_y(y, final_layer)

        self.epoch = 0

        self.weights = self.initializeWeights(d, final_layer)

        

        for i in range(self.numEpochs):
            self.epoch = i

            gradient = self.initializeZeroWeights(d, final_layer)

            all_predictions = []
            for j in range(n):
                output = self.forwardPropogation(X[j])

                error = [None] * (len(self.layers) + 1)
                error[-1] = output[-1] - adjusted_y[j]
                
                #Compute error
                for k in range(len(self.layers) - 1, -1, -1):
                    current_theta = self.weights[k + 1]
                    gPrime_output = self.gPrime(output[k + 1])
                    error[k] = np.multiply((current_theta.T.dot(error[k + 1])[:-1]), gPrime_output)

                #Compute gradients
                for k in range(len(self.layers) + 1):
                    gradient[k] = gradient[k] + np.asmatrix(error[k]).T * np.asmatrix(np.append(output[k], 1))

                all_predictions.append(output[-1])


            #Average gradient
            for j in range(len(self.layers) + 1):
                if j == 0:
                    gradient[j] = gradient[j]/n
                else:
                    gradient[j] = gradient[j]/n + gradient[j]*self.regulization


            #Update weights
            for j in range(len(self.weights)):
                increment = np.multiply(gradient[j], self.learningRate)
                self.weights[j] = np.asarray(self.weights[j] - increment)

            # print "New cost is : " + str(self.costFunction(all_predictions, adjusted_y))


    def gPrime(self, z):
        n = z.size
        output = np.empty(n)
        for i in range(n):
            output[i] = z[i] * (1 - z[i])

        return output

    def sigmoid(self, z):
        n = z.size
        output = np.empty(n)
        for i in range(n):
            output[i] = 1 / (1 + np.exp(-z[i]))

        return output


    def forwardPropogation(self, X):
        ''' Takes one input value and returns the result of each layer '''

        previous_output = X
        current_output = None
        result = []
        result.append(X)
        for layer in range(len(self.weights)):
            previous_size = 0
            if layer == 0:
                d = X.size
                previous_size = d + 1
            else:
                previous_size = self.layers[layer - 1] + 1
            
            current_theta = self.weights[layer]

            with_1 = np.append(previous_output, [1], axis=1)

            current_output = np.asarray(current_theta) * np.asarray(with_1)

            current_output = np.sum(current_output, axis=1)

            previous_output = self.sigmoid(current_output)

            result.append(previous_output)

        return result

    def costFunction(self, y_predicted, actual_y):
        total_cost = 0
        num_samples = len(y_predicted)
        num_outputs = len(y_predicted[0])

        for i in range(num_samples):

            for j in range(num_outputs):
                if actual_y[i][j] == 1:
                    total_cost += np.log(y_predicted[i][j])
                else:
                    total_cost += np.log(1 - y_predicted[i][j])
                if math.isnan(total_cost):
                    pdb.set_trace()
        
        total_cost = -1 * total_cost / num_samples

        running_sum = 0
        for i in range(len(self.weights)):
            running_sum += np.sum(np.square(self.weights[i]))
        regularization = self.regulization/(2 * num_samples) * running_sum

        return total_cost + regularization


    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-dimensional numpy array of the predictions
        '''
        n, d = X.shape
        result = []
        for i in range(n):
            output = self.forwardPropogation(X[i])
            result.append(output[-1])

        final = np.empty(n)
        for i in range(n):
            final[i] = np.argmax(result[i])

        return final

    
    def visualizeHiddenNodes(self, filename):
        '''
        CIS 519 ONLY - outputs a visualization of the hidden layers
        Arguments:
            filename - the filename to store the image
        '''
        