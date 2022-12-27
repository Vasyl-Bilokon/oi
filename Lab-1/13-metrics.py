import tensorflow.compat.v1 as tf
import numpy as np

tf.disable_v2_behavior()

# Placeholders
x = tf.placeholder(tf.int32, [5])
y = tf.placeholder(tf.int32, [5])

# Metrics
acc, acc_op = tf.metrics.accuracy(labels=x, predictions=y)

# Session
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

# Variable
val = sess.run([acc, acc_op], feed_dict={x: [1, 1, 0, 1, 0], y: [0, 1, 0, 0, 1]})

# Display results
val_acc = sess.run(acc)
print(val_acc)
