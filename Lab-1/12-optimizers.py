import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

# Variables definition
x = tf.Variable(3, name='x', dtype=tf.float32)
log_x = tf.log(x)
log_x_squared = tf.square(log_x)

# Optimizer
optimizer = tf.train.GradientDescentOptimizer(0.7)
train = optimizer.minimize(log_x_squared)

# Initialization
init = tf.global_variables_initializer()

# Computing
with tf.Session() as session:
    session.run(init)
    print("starting at", "x: ", session.run(x), "log(x)^2: ", session.run(log_x_squared))
    for step in range(10):
        session.run(train)
        print("step", step, "x: ", session.run(x), "log(x)^2: ", session.run(log_x_squared))
