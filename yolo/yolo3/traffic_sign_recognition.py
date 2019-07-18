from collections import namedtuple
import tensorflow as tf

num_classes = 58
Parameters = namedtuple('Parameters', [
    # Data parameters
    'num_classes', 'image_size',
    # Training parameters
    'batch_size', 'max_epochs', 'log_epoch', 'print_epoch',
    # Optimisations
    'learning_rate_decay', 'learning_rate',
    'l2_reg_enabled', 'l2_lambda',
    'early_stopping_enabled', 'early_stopping_patience',
    'resume_training',
    # Layers architecture
    'conv1_k', 'conv1_d', 'conv1_p',
    'conv2_k', 'conv2_d', 'conv2_p',
    'conv3_k', 'conv3_d', 'conv3_p',
    'fc4_size', 'fc4_p'
])


def fully_connected(input, size):
    weights = tf.get_variable('weights',
                              shape=[input.get_shape()[1], size],
                              initializer=tf.contrib.layers.xavier_initializer()
                              )
    biases = tf.get_variable('biases',
                             shape=[size],
                             initializer=tf.constant_initializer(0.0)
                             )
    return tf.matmul(input, weights) + biases


def fully_connected_relu(input, size):
    return tf.nn.relu(fully_connected(input, size))


def conv_relu(input, kernel_size, depth):
    weights = tf.get_variable('weights',
                              shape=[kernel_size, kernel_size, input.get_shape()[3], depth],
                              initializer=tf.contrib.layers.xavier_initializer()
                              )
    biases = tf.get_variable('biases',
                             shape=[depth],
                             initializer=tf.constant_initializer(0.0)
                             )
    conv = tf.nn.conv2d(input, weights,
                        strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.relu(conv + biases)


def pool(input, size):
    return tf.nn.max_pool(
        input,
        ksize=[1, size, size, 1],
        strides=[1, size, size, 1],
        padding='SAME'
    )


def model_pass(input, params, is_training):
    """
    Performs a full model pass.

    Parameters
    ----------
    input         : Tensor
                    Batch of examples.
    params        : Parameters
                    Structure (`namedtuple`) containing model parameters.
    is_training   : Tensor of type tf.bool
                    Flag indicating if we are training or not (e.g. whether to use dropout).

    Returns
    -------
    Tensor with predicted logits.
    """
    # Convolutions

    with tf.variable_scope('conv1'):
        conv1 = conv_relu(input, kernel_size=params.conv1_k, depth=params.conv1_d)
        pool1 = pool(conv1, size=2)
        pool1 = tf.cond(is_training, lambda: tf.nn.dropout(pool1, keep_prob=params.conv1_p), lambda: pool1)
    with tf.variable_scope('conv2'):
        conv2 = conv_relu(pool1, kernel_size=params.conv2_k, depth=params.conv2_d)
        pool2 = pool(conv2, size=2)
        pool2 = tf.cond(is_training, lambda: tf.nn.dropout(pool2, keep_prob=params.conv2_p), lambda: pool2)
    with tf.variable_scope('conv3'):
        conv3 = conv_relu(pool2, kernel_size=params.conv3_k, depth=params.conv3_d)
        pool3 = pool(conv3, size=2)
        pool3 = tf.cond(is_training, lambda: tf.nn.dropout(pool3, keep_prob=params.conv3_p), lambda: pool3)

    # Fully connected

    # 1st stage output
    pool1 = pool(pool1, size=4)
    shape = pool1.get_shape().as_list()
    pool1 = tf.reshape(pool1, [-1, shape[1] * shape[2] * shape[3]])

    # 2nd stage output
    pool2 = pool(pool2, size=2)
    shape = pool2.get_shape().as_list()
    pool2 = tf.reshape(pool2, [-1, shape[1] * shape[2] * shape[3]])

    # 3rd stage output
    shape = pool3.get_shape().as_list()
    pool3 = tf.reshape(pool3, [-1, shape[1] * shape[2] * shape[3]])

    flattened = tf.concat(1, [pool1, pool2, pool3])

    with tf.variable_scope('fc4'):
        fc4 = fully_connected_relu(flattened, size=params.fc4_size)
        fc4 = tf.cond(is_training, lambda: tf.nn.dropout(fc4, keep_prob=params.fc4_p), lambda: fc4)
    with tf.variable_scope('out'):
        logits = fully_connected(fc4, size=params.num_classes)
    return logits


X = tf.placeholder(tf.float32, [None, 32, 32])
Y = tf.placeholder(tf.float32, [None, num_classes])

# Construct model
params = Parameters
params.num_classes = num_classes
params.image_size = [32, 32]
params.batch_size = 128
params.conv1_k = 5
params.conv1_d = 32
params.conv1_p = 0.9
params.conv2_d = 64
params.conv2_k = 5
params.conv2_p = 0.8
params.conv3_d = 128
params.conv3_k = 5
params.conv3_p = 0.7
params.fc4_size = 1024
params.fc4_p = 0.5
params.l2_lambda = 0.0001
params.early_stopping_enabled = True
params.early_stopping_patience = 100
params.learning_rate = 0.001

logits = model_pass(X, params, True)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
train_op = optimizer.minimize(loss_op)


# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, num_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.8})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y,
                                                                 keep_prob: 1.0})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for 256 MNIST test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: mnist.test.images[:256],
                                      Y: mnist.test.labels[:256],
keep_prob: 1.0}))