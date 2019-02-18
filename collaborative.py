import re
import time
import random
import numpy as np 
import pandas as pd 
import tensorflow as tf 
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

df = pd.read_csv('ml-1m/ratings.dat', sep='\t', names=['user', 'item', 'rating', 'timestamp'], header=None)
df = df.drop('timestamp', axis=1)
num_items = df.item.nunique()
num_users = df.user.nunique()
print("USERS: {} ITEMS: {}".format(num_users, num_items))



X_train ,X_test = train_test_split(df,test_size=0.2)
print X_train.shape
print X_test.shape

r = X_train['rating'].values.astype(float)
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(r.reshape(-1,1))
df_normalized = pd.DataFrame(x_scaled)
X_train['rating'] = df_normalized
print(X_train.head(5))

matrix = X_train.pivot(index='user', columns='item', values='rating')
matrix.fillna(0, inplace=True)
print(matrix.head(5))

users = matrix.index.tolist()
items = matrix.columns.tolist()

matrix = matrix.as_matrix()

num_input = matrix.shape[1]
num_hidden_1 = 10
num_hidden_2 = 5

X = tf.placeholder(tf.float64, [None, num_input])


weights = {
	'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1], dtype=tf.float64)),
	'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2], dtype=tf.float64)),
	'decoder_h1': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1], dtype=tf.float64)),
	'decoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_input], dtype=tf.float64)),
}

biases = {
	'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1], dtype=tf.float64)),
	'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2], dtype=tf.float64)),
	'decoder_b1': tf.Variable(tf.random_normal([num_hidden_1], dtype=tf.float64)),
	'decoder_b2': tf.Variable(tf.random_normal([num_input], dtype=tf.float64)),
}

def encoder(x):
	# Encoder Hidden layer with sigmoid activation #1
	layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
	# Encoder Hidden layer with sigmoid activation #2
	layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
	return layer_2


# Building the decoder

def decoder(x):
	# Decoder Hidden layer with sigmoid activation #1
	layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
	# Decoder Hidden layer with sigmoid activation #2
	layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))
	return layer_2


# Construct model

encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction

y_pred = decoder_op


# Targets are the input data.

y_true = X
# Define loss and optimizer, minimize the squared error

loss = tf.losses.mean_squared_error(y_true, y_pred)
optimizer = tf.train.RMSPropOptimizer(0.03).minimize(loss)

predictions = pd.DataFrame()

# Define evaluation metrics

eval_x = tf.placeholder(tf.int32, )
eval_y = tf.placeholder(tf.int32, )
pre, pre_op = tf.metrics.precision(labels=eval_x, predictions=eval_y)
init = tf.global_variables_initializer()
local_init = tf.local_variables_initializer()

with tf.Session() as session:
	epochs = 20
	batch_size = 256

	session.run(init)
	session.run(local_init)
	print matrix.shape
	num_batches = int(matrix.shape[0] / batch_size)
	matrix = np.array_split(matrix, num_batches)
	#print matrix.shape[0:1]
	print "*************************"

	for i in range(epochs):

		avg_cost = 0

		for batch in matrix:
			_, l = session.run([optimizer, loss], feed_dict={X: batch})
			avg_cost += l

		avg_cost /= num_batches

		print("Epoch: {} Loss: {}".format(i + 1, avg_cost))

	print("Predictions...")

	matrix = np.concatenate(matrix, axis=0)

	preds = session.run(decoder_op, feed_dict={X: matrix})

	predictions = predictions.append(pd.DataFrame(preds))

	predictions = predictions.stack().reset_index(name='rating')
	predictions.columns = ['user', 'item', 'rating']
	predictions['user'] = predictions['user'].map(lambda value: users[value])
	predictions['item'] = predictions['item'].map(lambda value: items[value])
	print("Filtering out items in training set")

	keys = ['user', 'item']
	i1 = predictions.set_index(keys).index
	i2 = df.set_index(keys).index

	recs = predictions[~i1.isin(i2)]
	recs = recs.sort_values(['user', 'rating'], ascending=[True, False])
	recs = recs.groupby('user').head(10)
	recs.to_csv('recs.tsv', sep='\t', index=False, header=False)


