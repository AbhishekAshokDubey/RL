import tensorflow as tf
import tensorflow.contrib.slim as slim

class neural_network():
    def __init__(self, D, H, learning_rate):
        self.input_layer = tf.placeholder(shape=[None,D],dtype=tf.float32)
        self.hidden_layer = slim.fully_connected(self.input_layer, H, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), biases_initializer=None)
        self.output_layer = slim.fully_connected(self.hidden_layer, 1, activation_fn=tf.nn.sigmoid, weights_initializer=tf.contrib.layers.xavier_initializer(), biases_initializer=None)

        self.actual_actions = tf.placeholder(shape=[None],dtype=tf.int32)
        self.game_rewards = tf.placeholder(shape=[None,1],dtype=tf.float32)     
        self.comparison = tf.equal(self.actual_actions,tf.constant(1))
        
        # loss_fn = Ya(log(Yp)) + (1-Ya)log(1-Yp)
        self.log_loss = tf.where(self.comparison, tf.log(self.output_layer), tf.log(tf.subtract(1.0,self.output_layer)))
        self.policy_loss = -tf.reduce_mean(self.log_loss * self.game_rewards)

        self.w_variables = tf.trainable_variables()
        self.opt = tf.train.RMSPropOptimizer(decay = decay_rate, learning_rate=learning_rate)
		
        self.train = self.opt.minimize(self.policy_loss)
		
        # below code for accumulating grads and applying after n steps
        self.accum_grad_variables = [tf.Variable(tf.zeros_like(w), trainable=False) for w in self.w_variables]
        self.grads_and_vars = self.opt.compute_gradients(self.policy_loss, self.w_variables)
        self.accum_grad_ops = [self.accum_grad_variables[i].assign_add(grad_and_var[0]) for i, grad_and_var in enumerate(self.grads_and_vars)]
        self.train_step = self.opt.apply_gradients([(self.accum_grad_variables[i], grad_and_var[1]) for i, grad_and_var in enumerate(self.grads_and_vars)])
        self.reset_accum_grads_variables_ops = [w.assign(tf.zeros_like(w)) for w in self.accum_grad_variables]
        self.saver = tf.train.Saver()
