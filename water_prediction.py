"""
Water Usage Forecasting using RNN (Recurrent Neural Network)

Samruddhi Taywade 
Final Year B.E. 
P.R.Patil College of Engineering and Technology

"""



#Import Libraries
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import random
#%matplotlib inline
import tensorflow as tf
import shutil
import tensorflow.contrib.learn as tflearn
import tensorflow.contrib.layers as tflayers
from tensorflow.contrib.learn.python.learn import learn_runner
import tensorflow.contrib.metrics as metrics
import tensorflow.contrib.rnn as rnn

from numpy import sin, cos
import scipy.integrate as integrate
import matplotlib.animation as animation

data = pd.read_csv('water_data.csv',index_col='Day')

#plot of water used in last three months
data[:].plot()
plt.xlabel('Days')
plt.ylabel('Water Used in (litres)')
plt.title('Pot of Water Used in last  month')
plt.show()

ts = np.array(data['Water_Used']).reshape((-1,1))


#Convert data into array that can be broken up into training "batches" that we will feed into our RNN model.  Note the shape of the arrays. 

TS = np.array(ts)
num_periods = 7
f_horizon = 1  #forecast horizon, one period into the future

x_data = TS[:(len(TS)-(len(TS) % num_periods))]
#print(len(x_data))
#print("x_data : ",x_data)
x_batches = x_data.reshape(-1, 7, 1)

#print (len(x_batches))
print ("x_batches shape : ",x_batches.shape)
#print (x_batches[0:1])

y_data = TS[1:(len(TS)-(len(TS) % num_periods))+f_horizon]
#print(y_data.shape)
y_batches = y_data.reshape(-1, 7, 1)


#Pull out our test data

def test_data(series,forecast,num_periods):
    test_x_setup = TS[-(num_periods + forecast):]
    testX = test_x_setup[:num_periods].reshape(-1, 7, 1)
    testY = TS[-(num_periods):].reshape(-1, 7, 1)
    return testX,testY

X_test, Y_test = test_data(TS,f_horizon,num_periods )
print (X_test.shape)
print(len(X_test))
#print (X_test)

print (Y_test.shape)
print(len(Y_test))
#print (Y_test)
#print ("y_batches : ",y_batches[0:1])
print ("y_batches shape : ",y_batches.shape)


#### RNN MODEL #######

tf.reset_default_graph()   #We didn't have any previous graph objects running, but this would reset the graphs

num_periods = 7      #number of periods per vector we are using to predict one period ahead
inputs = 1            #number of vectors submitted
hidden = 7          #number of neurons we will recursively work through, can be changed to improve accuracy
output = 1            #number of output vectors

X = tf.placeholder(tf.float32, [None, num_periods, inputs])   #create variable objects
y = tf.placeholder(tf.float32, [None, num_periods, output])

print(X)


basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden, activation=tf.nn.relu)   #create our RNN object
rnn_output, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)               #choose dynamic over static

learning_rate = 0.001   #small learning rate so we don't overshoot the minimum

stacked_rnn_output = tf.reshape(rnn_output, [-1, hidden])           #change the form into a tensor
stacked_outputs = tf.layers.dense(stacked_rnn_output, output)        #specify the type of layer (dense)
outputs = tf.reshape(stacked_outputs, [-1, num_periods, output])          #shape of results
 
loss = tf.reduce_sum(tf.square(outputs - y))    #define the cost function which evaluates the quality of our model
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)          #advanced gradient descent method 
training_op = optimizer.minimize(loss)          #train the result of the application of the cost_function                                 

init = tf.global_variables_initializer()           #initialize all the variables


with tf.Session() as sess:
    writer = tf.summary.FileWriter("water_output", sess.graph)
    print(sess.run(init))
    writer.close()

epochs = 10000     #number of iterations or training cycles, includes both the FeedFoward and Backpropogation
errors = []
iterations = []
with tf.Session() as sess:
    init.run()
    for ep in range(epochs):
        sess.run(training_op, feed_dict={X: x_batches, y: y_batches})
        errors.append(loss.eval(feed_dict={X: x_batches, y: y_batches}))
        iterations.append(ep)
        if ep % 100 == 0:
            mse = loss.eval(feed_dict={X: x_batches, y: y_batches})
            print(ep, "\tMSE:", mse)  
    y_pred = sess.run(outputs, feed_dict={X: X_test})
    #print(y_pred)
    
print('Actual Data \t Predicted Data')
for i in range(len(y_pred[0])):
    print(Y_test[0][i] , y_pred[0][i])


plt.title("Forecast vs Actual", fontsize=14)
plt.plot(pd.Series(np.ravel(Y_test)), "bo", markersize=10, label="Actual")
#plt.plot(pd.Series(np.ravel(Y_test)), "w*", markersize=10)
X = "Sun Mon Tues Weds Thurs Fri Sat ".split()
#X = ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday']
plt.plot(X,pd.Series(np.ravel(y_pred)), "r.", markersize=10, label="Forecast")
plt.legend()
plt.xlabel("Next Week")
plt.show()


error = []
print(y_pred[0][1][0])
print(abs(y_pred[0][0][0]-Y_test[0][0][0]))
print(len(y_pred[0]))
for i in range(len(y_pred[0])):
    err = abs((y_pred[0][i][0]-Y_test[0][i][0])/Y_test[0][i][0])
    error.append(err)
x = np.arange(len(error))

plt.bar(x,error,align='center')
plt.xlabel('predicted points')
plt.ylabel('mean square error')
plt.title('mean square error histogram >>>')
plt.show()



errors=np.array(errors)
iterations=np.array(iterations)
print(errors.shape)
#print(errors)

plt.hist(errors,iterations,label='errors', facecolor='orange')

plt.xlabel('iterations')
plt.ylabel('mean square error ')
plt.title('histogram of errors during training -->>>>>')
plt.legend()
plt.show()



errors=np.array(errors)
iterations=np.array(iterations)
print(errors.shape)
#print(errors)

plt.plot(errors,iterations,label='errors',color='green')

plt.xlabel('iterations')
plt.ylabel('mean square error ')
plt.title('plot of errors during training-->>>>>')
plt.legend()
plt.show()

