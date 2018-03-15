from data_digit_recog import *
from models_digit_recog import *
NUM_EPOCH=20
BATCH_SIZE=500


################################################################### MULTI LAYER PERCEPTRON #####################################################
print 'Running MLP Model....'
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#Images are in 3D format consisting instance_no,image_width,image_height
#For MLP, we need to convert into a vector error array. Here we convert to 28pixels*28pixels =784 input values

num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')
X_train=scale_data(X_train)
X_test=scale_data(X_test)
y_train,y_test,num_classes=onehot_encode_outputs(y_train,y_test)


model = mlp_model(num_pixels,num_classes)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=NUM_EPOCH, batch_size=BATCH_SIZE, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("MLP Error: %.2f%%" % (100-scores[1]*100))





################################################################### CONVOLUTIONAL NEURAL NETWORK ################################################
print 'Running CNN Model....'

(X_train, y_train), (X_test, y_test) = mnist.load_data()
# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')
X_train=scale_data(X_train)
X_test=scale_data(X_test)

y_train,y_test,num_classes=onehot_encode_outputs(y_train,y_test)

# build the model
model = cnn_model(num_classes)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=NUM_EPOCH, batch_size=BATCH_SIZE, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))

