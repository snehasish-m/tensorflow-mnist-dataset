# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 11:19:20 2017

@author: Snehasish
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

data = input_data.read_data_sets("/tmp/data/", one_hot = True)

#learning rate
LR = 0.001

#respective size of hidden layers
hl1 = 500  
hl2 = 500
hl3 = 500

output_size = 10  #{one-hot encoded array indicating any digit}
batch_size = 100
no_of_epochs = 10
total_no_of_data = data.train.num_examples

#for testing data
test_x = data.test.images
test_y = data.test.labels

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def NNmodel(data):
    #initialize weights and biases of each layer with random tensorflow variables.
    h_layer1 = {'weights':tf.Variable(tf.random_normal([784, hl1])), 'biases':tf.Variable(tf.random_normal([hl1]))}

    h_layer2 = {'weights':tf.Variable(tf.random_normal([hl1, hl2])), 'biases':tf.Variable(tf.random_normal([hl2]))}

    h_layer3 = {'weights':tf.Variable(tf.random_normal([hl2, hl3])), 'biases':tf.Variable(tf.random_normal([hl3]))}
                                                       
    output_layer = {'weights':tf.Variable(tf.random_normal([hl3, output_size])), 'biases':tf.Variable(tf.random_normal([output_size]))}
     
    #  (weights1 * weights2) + biases    (using rectified liniear activation)                             
                     
    l1 = tf.add(tf.matmul(data,h_layer1['weights']), h_layer1['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,h_layer2['weights']), h_layer2['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,h_layer3['weights']), h_layer3['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']
                       
    return output

def train_model(x):
    #predicting output for the data
    pred = NNmodel(x)
    
    #error after predicting the digit
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y) )
    
    #optimizer to reduce the error term (or cost)
    optimizer = tf.train.AdamOptimizer(learning_rate= LR).minimize(cost)
    
    # upto this point the model is only defined, but to run the model  
    # we have to run it within a session.

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for ep in range(no_of_epochs):
            epoch_loss = 0
            for _ in range(int(total_no_of_data/batch_size)):
                #batch-wise data is trained
                ep_x, ep_y = data.train.next_batch(batch_size)
                
                #cost(c) for this batch is calaculated
                _, c = sess.run([optimizer, cost], feed_dict={x: ep_x, y: ep_y})
                epoch_loss += c

            print('Epoch: ', ep+1, '/',no_of_epochs,'loss:',epoch_loss)
        
        #no of correct predictions
        correct = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        
        #calculating the final accuracy
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: test_x, y: test_y })*100, '%' )

    
train_model(x)