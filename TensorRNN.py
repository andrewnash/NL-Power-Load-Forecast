import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
#%matplotlib inline

df = pd.read_excel("jan2017.xlsx","FromQuery")
df.head()

# a = df['ActualIslandTotal'].values
# b = df['ActualStJohnsTemp'].values
# y = df['ActualIslandTotal'].values
# y = df['ActualIslandTotal'].values
temp = df['ActualStJohnsTemp'].values
wind = df['ActualStJohnsWind'].values
cloud = df['ActualStJohnsCloud'].values
power = df['ActualIslandTotal'].values
power_tminus1 = np.insert(power, 0, 1171.5176593) #sift values forward
power_tminus1 = power_tminus1[:-1] #delete extra power at end

data = np.column_stack((temp,wind,cloud,power_tminus1))

print('Total number of hours in the dataset: {}'.format(len(power)))

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data.reshape(-1, 1))
scaled_power = scaler.fit_transform(power.reshape(-1, 1))
plt.figure(figsize=(12,7), frameon=False, facecolor='brown', edgecolor='blue')
plt.title('NL Island Power 2017')
plt.xlabel('Hours')
plt.ylabel('Scaled Power')
plt.plot(scaled_power, label='Power Data(MW)')
plt.legend()
plt.show()

def window_data(data, window_size):
    X = []
    y = []
    
    i = 0
    L = len(data)
    if len(data) > 10000:
    	L = L/3

    while (i + window_size) <= L - 1:
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
        
        i += 1
    assert len(X) ==  len(y)
    return X, y

X, y = window_data(scaled_data, 7)

X_train  = np.array(X[:7400])
X_test = np.array(X[7400:])

print(len(X_test))

X, y = window_data(scaled_power, 7)

y_train = np.array(y[:7400])
y_test = np.array(y[7400:])


# X_train = np.column_stack((X_train,y_train))
# X_test = np.column_stack((X_test,y_test))


print("X_train size: {}".format(X_train.shape))
print("y_train size: {}".format(y_train.shape))
print("X_test size: {}".format(X_test.shape))
print("y_test size: {}".format(y_test.shape))

epochs = 200
batch_size = 24

def LSTM_cell(hidden_layer_size, batch_size,number_of_layers, dropout=True, dropout_rate=0.8):
    
    layer = tf.contrib.rnn.BasicLSTMCell(hidden_layer_size)
    
    if dropout:
        layer = tf.contrib.rnn.DropoutWrapper(layer, output_keep_prob=dropout_rate)
        
    cell = tf.contrib.rnn.MultiRNNCell([layer]*number_of_layers)
    
    init_state = cell.zero_state(batch_size, tf.float32)
    
    return cell, init_state

def output_layer(lstm_output, in_size, out_size):
    
    x = lstm_output[:, -1, :]
    print(x)
    weights = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=0.05), name='output_layer_weights')
    bias = tf.Variable(tf.zeros([out_size]), name='output_layer_bias')
    
    output = tf.matmul(x, weights) + bias
    return output

def opt_loss(logits, targets, learning_rate, grad_clip_margin):
    
    losses = []
    for i in range(targets.get_shape()[0]):
        losses.append([(tf.pow(logits[i] - targets[i], 2))])
        
    loss = tf.reduce_sum(losses)/(2*batch_size)
    
    #Cliping the gradient loss
    gradients = tf.gradients(loss, tf.trainable_variables())
    clipper_, _ = tf.clip_by_global_norm(gradients, grad_clip_margin)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_optimizer = optimizer.apply_gradients(zip(gradients, tf.trainable_variables()))
    return loss, train_optimizer

class LoadForecastRNN(object):
    
    def __init__(self, learning_rate=0.001, batch_size=24, hidden_layer_size=512, number_of_layers=1, 
                 dropout=True, dropout_rate=0.8, number_of_classes=1, gradient_clip_margin=4, window_size=7):
    
        self.inputs = tf.placeholder(tf.float32, [batch_size, window_size, 1], name='input_data')
        self.targets = tf.placeholder(tf.float32, [batch_size, 1], name='targets')

        cell, init_state = LSTM_cell(hidden_layer_size, batch_size, number_of_layers, dropout, dropout_rate)

        outputs, states = tf.nn.dynamic_rnn(cell, self.inputs, initial_state=init_state)

        self.logits = output_layer(outputs, hidden_layer_size, number_of_classes)

        self.loss, self.opt = opt_loss(self.logits, self.targets, learning_rate, gradient_clip_margin)

tf.reset_default_graph()
model = LoadForecastRNN()

session =  tf.Session()
session.run(tf.global_variables_initializer())

for i in range(epochs):
    traind_scores = []
    ii = 0
    epoch_loss = []
    while(ii + batch_size) <= len(X_train):
        X_batch = X_train[ii:ii+batch_size]
        y_batch = y_train[ii:ii+batch_size]
        
        o, c, _ = session.run([model.logits, model.loss, model.opt], feed_dict={model.inputs:X_batch, model.targets:y_batch})
        
        epoch_loss.append(c)
        traind_scores.append(o)
        ii += batch_size
    if (i % 30) == 0:
        print('Epoch {}/{}'.format(i, epochs), ' Current loss: {}'.format(np.mean(epoch_loss)))


sup =[]
for i in range(len(traind_scores)):
    for j in range(len(traind_scores[i])):
        sup.append(traind_scores[i][j])

tests = []
i = 0
while i+batch_size <= len(X_test):
    
    o = session.run([model.logits], feed_dict={model.inputs:X_test[i:i+batch_size]})
    i += batch_size
    tests.append(o)

tests_new = []
for i in range(len(tests)):
    for j in range(len(tests[i][0])):
        tests_new.append(tests[i][0][j])

test_results = []
for i in range(735):
    if i >= 651:
        test_results.append(tests_new[i-651])
    else:
        test_results.append(None)

plt.figure(figsize=(16, 7))
plt.plot(scaled_datasetB, label='Original data')
plt.plot(sup, label='Training data')
plt.plot(test_results, label='Testing data')
plt.legend()
plt.show()

session.close()
