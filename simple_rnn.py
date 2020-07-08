import tensorflow as tf
import numpy as np

n_inputs = 3
n_neurons = 5
n_steps = 2
batch_size = 4

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])


basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

Y = outputs
final_output = Y[:,-1,:]

init = tf.global_variables_initializer()


# Mini-batch:        instance 0,instance 1,instance 2,instance 3
#Vector of integers as input
#batch of 4
#Sequence length of 2
X_batch = np.array([
         # t = 0     t = 1
        [[0, 1, 2], [9, 8, 7]], # instance 0
        [[3, 4, 5], [0, 0, 0]], # instance 1
        [[6, 7, 8], [6, 5, 4]], # instance 2
        [[9, 0, 1], [3, 2, 1]], # instance 3
    ])

X_batch = np.random.randint(0,10,size=(batch_size,n_steps,n_inputs))

with tf.Session() as sess:
    init.run()
    Y_val, X_out,fin_out = sess.run([Y,X,final_output], feed_dict={X: X_batch})


print(Y_val)
print(np.shape(Y_val))
print(np.shape(X_out))
print(fin_out)
print(np.shape(fin_out))
print('end')