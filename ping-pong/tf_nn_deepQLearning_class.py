#ref: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

import tensorflow as tf
import tensorflow.contrib.slim as slim

#tf.reset_default_graph()

def pool_layer(in_layer):
    return tf.layers.max_pooling2d(inputs=in_layer, pool_size=[2, 2], strides=(2,2), padding="same")

# replaces model initializations, forward_prop and back_prop
class neural_network():
    def __init__(self, img_dim_x = 80, img_dim_y = 80, farmes = 4, action_count = 2, DL = 256, learning_rate=1e-3):

	self.input_screen = tf.placeholder(shape=[None,img_dim_x, img_dim_y, farmes],dtype=tf.float32)
        #self.action_taken is same as self.max_rew_action below,
        #but its here to avoid the mutiple compute of the same
        self.action_taken = tf.placeholder(shape=[None],dtype=tf.int32)
        self.updated_q_value = tf.placeholder(shape=[None],dtype=tf.float32) # for the taken action
	
	self.input_layer = tf.reshape(self.input_screen, [-1, img_dim_x, img_dim_y, farmes])
        self.conv1 = tf.layers.conv2d(inputs=self.input_layer, filters=32, kernel_size=[8, 8], strides=(4,4), padding="same", activation=tf.nn.relu)
        self.pool1 = pool_layer(self.conv1)
        self.conv2 = tf.layers.conv2d(inputs=self.pool1, filters=64, kernel_size=[4, 4], strides=(2,2), padding="same", activation=tf.nn.relu)
        self.pool2 = pool_layer(self.conv2)
        self.conv3 = tf.layers.conv2d(inputs=self.pool2, filters=64, kernel_size=[3, 3], strides=(1,1), padding="same", activation=tf.nn.relu)
        self.pool3 = pool_layer(self.conv3)
        self.flatten_layer = tf.contrib.layers.flatten(self.pool3)
        self.dense_connected_layer = slim.fully_connected(self.flatten_layer, DL, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), biases_initializer=None)
        self.output_layer = slim.fully_connected(self.dense_connected_layer, action_count, activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer(), biases_initializer=None)
        self.max_rew_action = tf.argmax(self.output_layer,1)

        # feel free to use tf.gather instead of tf.one_hot
        self.action_taken_onehot = tf.one_hot(indices=self.action_taken, depth=action_count, on_value=1.0, off_value=0.0, axis=-1)
        self.old_q_value = tf.reduce_sum(tf.multiply(self.output_layer, self.action_taken_onehot), reduction_indices = 1)

        self.loss = tf.reduce_sum(tf.square(self.updated_q_value - self.old_q_value))
        
        self.w_variables = tf.trainable_variables()
        self.opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.train = self.opt.minimize(self.loss)

        # below code for accumulating grads and applying after n steps
        self.accum_grad_variables = [tf.Variable(tf.zeros_like(w.initialized_value()), trainable=False) for w in self.w_variables]
        self.grads_and_vars = self.opt.compute_gradients(self.loss, self.w_variables)
        #self.grads_and_vars = tf.gradients(self.loss, self.w_variables)
        self.accum_grad_ops = [self.accum_grad_variables[i].assign_add(grad_and_var[0], use_locking=False) for i, grad_and_var in enumerate(self.grads_and_vars)]
        self.train_step = self.opt.apply_gradients([(self.accum_grad_variables[i], grad_and_var[1]) for i, grad_and_var in enumerate(self.grads_and_vars)])
        self.reset_accum_grads_variables_ops = [w.assign(tf.zeros_like(w)) for w in self.accum_grad_variables]
        self.saver = tf.train.Saver()
