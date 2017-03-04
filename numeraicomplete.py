import pandas as pd 
import numpy as np 
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score as AUC, accuracy_score as accuracy, log_loss
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Reshape, Permute, Flatten
from keras.layers.convolutional import MaxPooling2D, ZeroPadding1D, MaxPooling1D,Convolution1D
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import GRU, LSTM
#from keras.optimizers import SGD
from matplotlib import pyplot as plt
import os


def getdata(input_file):
	print('Getting the file')
	df = pd.read_csv(input_file)
	x_train = df.iloc[:,0:-1]
	y_train = df.iloc[:,-1] 
	print(x_train.shape)
	print(y_train.shape)   
	return x_train, y_train

def create_model(model_type):
	if(model_type=='cnn'):
		input_shape=(21)
		model = Sequential()
		
		#model.add(BatchNormalization())
		#model.add(ZeroPadding1D(40))
		
		model.add(Convolution1D(256,3,activation='relu',input_shape=(21,1)))
		model.add(Convolution1D(256,3,activation='relu'))
		model.add(MaxPooling1D(2))
		model.add(Dropout(0.2))

		model.add(Convolution1D(128,3,activation='relu'))
		model.add(Convolution1D(128,3,activation='relu'))
		model.add(MaxPooling1D(2))
		model.add(Dropout(0.2))

		model.add(LSTM(128))
		#model.add(LSTM(64))
		#model.add(Convolution1D(64,3,activation='relu'))
		#model.add(Convolution1D(64,3,activation='relu'))
		#model.add(MaxPooling1D(2))
		#model.add(Dropout(0.2))
		#model.summary()
		#model.add(Flatten())
		model.add(Dense(1))
		model.compile(loss='mean_squared_error', optimizer='nadam', metrics=['accuracy'])
		return model

def train_and_evaluate(input_file,test_file,output_file, save_model=True):
	x_train,y_train=getdata(input_file)
	x_train=np.expand_dims(x_train,-1)
	x_train2=np.expand_dims(x_train,1)
	print(x_train.shape)
	model_type='cnn'
	model=create_model(model_type=model_type)
	model.summary()

	cur_iter=100
	model_filename = model_type + str(cur_iter)
	if os.path.isfile(model_filename):
		print('Loading weights from file')
		model.load_weights(model_filename)
	'''
	num_iters = 100
	epochs_per_iter = 10
	batch_size = 1000

	print('Training will start now...')
	#create a loop for iteration and save each iteration with small number of training epochs
	#then try to load weights from previous epochs and run
	while cur_iter < num_iters:
		print('Iteration: ' + str(cur_iter))

		history=model.fit(x_train,y_train, validation_split=0.2, nb_epoch=epochs_per_iter, batch_size=batch_size, verbose=1)
		#increase number of epochs done
		cur_iter += epochs_per_iter

	#list all data in history
	print(history.history.keys())
	#display model evaluation
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('Final model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()
	# summarize history for loss
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('Final model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()
	
	#save the model midway
	model_json = model.to_json()
	with open(model_type + str(cur_iter), "w") as json_file:
		json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights(model_type + str(cur_iter))
	print("Saved model to disk")

	'''
	test = pd.read_csv(test_file)
	x_test = test.drop( 't_id', axis = 1)
	x_test=np.expand_dims(x_test,-1)
	print('Predicting...')
	predictions = model.predict_proba(x_test)
	print('Saving...')
	print(predictions.shape)
	test['probability'] = predictions[:,1]
	test.to_csv( output_file, columns = ( 't_id', 'probability' ), index = None )

input_file = 'numerai_training_data.csv'
test_file = 'numerai_tournament_data.csv'
output_file = 'predictions_lr.csv'
train_and_evaluate(input_file, test_file,output_file, save_model=False)
