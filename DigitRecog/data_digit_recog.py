from keras.datasets import mnist	#get the dataset
import matplotlib.pyplot as plt  	#get the plot library
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
import numpy

#################################################### Checking the Data ##################################################

# load (downloaded if needed) the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()



#Creating sample plot
plt.subplot(221)
plt.imshow(X_train[0])
plt.subplot(222)
plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))

# show the plot
#plt.show()

#print len(X_train) #60K samples in Training dataset
#print len(X_test)  #10K samples in Testing dataset


seed = 3 #setting up random seed
numpy.random.seed(seed)	




def scale_data(X):
	#scaling  values(between 0-255) to 0-1
	return X / 255

def onehot_encode_outputs(y_train,y_test):
	y_train = np_utils.to_categorical(y_train)
	y_test = np_utils.to_categorical(y_test)
	num_classes = y_test.shape[1]
	return y_train,y_test,num_classes