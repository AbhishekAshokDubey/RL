""" Trains an agent with (stochastic) Q Learning on Pong. Uses OpenAI Gym. """
import numpy as np
#import cPickle as pickle
import gym
import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
#import matplotlib.pyplot as plt
import random

# hyperparameters
batch_size = 32 # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99 # discount factor for reward
randomness = 0.1 # initial prob to corrupt (randomness) the action
render = False

replay_memory_limit = 10000

save_freq = 10 # keep zero if you dun want to save model
D = (80,80)  # input dimensionality: 80x80 grid
last_n_frames = 4

k = 2 # select an action every kth frame and repeat for other frames

model_save_path = os.path.join(os.getcwd(),'model_QLearning','mymodel.ckpt')

class replay_memory():
    def __init__(self, limit, cleanup_count=100, sample_count = 32):
        self.memory = [];
        self.limit = limit;
        self.cleanup_count = cleanup_count
        self.sample_count = sample_count

    def append(self, tupple):
        if len(self.memory) <= self.limit:
            self.memory.append(tupple)
        else:
            self.memory.pop(0)
            self.memory.append(tupple)

    def sample(self, count=None):
        if count is None:
            count = self.sample_count
        if count > len(self.memory):
            count = len(self.memory)
        return random.sample(self.memory,count)

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
        self.conv1 = tf.layers.conv2d(inputs=self.input_layer, filters=16, kernel_size=[8, 8], strides=4, padding="same", activation=tf.nn.relu)
        self.conv2 = tf.layers.conv2d(inputs=self.conv1, filters=32, kernel_size=[4, 4], strides=2, padding="same", activation=tf.nn.relu)
        self.flatten_layer = tf.contrib.layers.flatten(self.conv2)
        #self.dim = tf.shape(self.flatten_layer)[1]
        self.dense_connected_layer = slim.fully_connected(self.flatten_layer, DL, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), biases_initializer=None)
        self.output_layer = slim.fully_connected(self.dense_connected_layer, action_count, activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer(), biases_initializer=None)
        self.max_rew_action = tf.argmax(self.output_layer,1)

        # feel free to use tf.gather instead of tf.one_hot
        self.action_taken_onehot = tf.one_hot(indices=self.action_taken, depth=action_count, on_value=1.0, off_value=0.0, axis=-1)
        self.old_q_value = tf.reduce_sum(tf.multiply(self.output_layer, self.action_taken_onehot), reduction_indices = 1)

        self.loss = tf.reduce_sum(tf.square(self.updated_q_value - self.old_q_value))

        self.w_variables = tf.trainable_variables()
        self.gradients = []
        for indx,w in enumerate(self.w_variables):
            w_holder_var = tf.placeholder(tf.float32,name="w_"+ str(indx))
            self.gradients.append(w_holder_var)
        
        self.all_gradients = tf.gradients(self.loss,self.w_variables)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.apply_grads = optimizer.apply_gradients(zip(self.gradients,self.w_variables))
        self.train = optimizer.minimize(self.loss)     
        self.saver = tf.train.Saver()

def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    #  I = env.reset() # Use this to verify, whats happening
    #  plt.imshow(I)
    I = I[35:195] # crop and keep only the play area
    I = I[::2,::2,0] # downsample by factor of 2, take every second row and column, and take only "R" component out of RGB image
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (but paddles, ball) just set to 1
    return I.astype(np.float) # convert to 1D array and return

def get_reset_values(observation, nn, sess):
    nn_input = np.dstack([prepro(observation)]* last_n_frames)
    actions = sess.run(nn.max_rew_action, feed_dict={nn.input_screen: np.expand_dims(nn_input,axis=0)})
    return nn_input, actions


def get_Q_values(nn, sample):
    updated_q = []
    # sample: [(nn_input, action, reward, nn_input_next, terminal), ...]
    next_nn_input = np.vstack([np.expand_dims(x[-2],axis=0) for x in sample])
    next_state_q_vaules = sess.run(nn.output_layer, feed_dict={nn.input_screen: next_nn_input})
    for x, q_next in zip(sample, next_state_q_vaules):
        if x[-1]:
            q = x[2]
        else:
            q = x[2] + (gamma * np.max(q_next))
        updated_q.append(q)
    return np.array(updated_q);


env = gym.make("Pong-v0")
running_reward = None
reward_sum = 0
episode_number = 0


tf.reset_default_graph()
nn = neural_network()
memory = replay_memory(limit=replay_memory_limit)

with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(model_save_path))
    if ckpt and ckpt.model_checkpoint_path:
        print("using the saved model")
        nn.saver.restore(sess, model_save_path)
    else:
        sess.run(tf.global_variables_initializer())
        print("making new model")

    observation = env.reset()
    nn_input, actions = get_reset_values(observation, nn, sess)
    while True:
        if render: env.render()
        action = actions[0]
        if np.random.uniform() < randomness: action = abs(1-action) # roll the dice!
        game_action_key = 2 if action else 3

        for _ in range(k):
            observation, reward, done, info = env.step(game_action_key)
            terminal = (reward != 0)
            reward_sum += reward
            nn_input_next = np.append(nn_input[:,:,1:],np.expand_dims(prepro(observation),axis=2),axis=2)
            memory.append((nn_input, action, reward, nn_input_next, terminal))
            nn_input = nn_input_next.copy()
            if terminal:
                break;

        mini_batch = memory.sample(count=32)
        xs = np.vstack([np.expand_dims(x[0],axis=0) for x in mini_batch])
        acns = np.array([x[1] for x in mini_batch])        
        qs = get_Q_values(nn, mini_batch)

        sess.run(nn.train, feed_dict={nn.input_screen: xs, nn.updated_q_value:qs, nn.action_taken: acns})
        
        if terminal:
            nn_input, actions = get_reset_values(observation, nn, sess)
        else:
            actions = sess.run(nn.max_rew_action, feed_dict={nn.input_screen: np.expand_dims(nn_input,axis=0)})

        if done: # an episode finished
            episode_number += 1
            print(episode_number)
            randomness = randomness / (((episode_number*1.0) / 100) + 10)
                
            if (save_freq and not(episode_number%save_freq)):
                print("saving the model ...")
                nn.saver.save(sess, model_save_path)

            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            if reward_sum != -21.0:
                print('resetting env. episode reward %f. running mean: %f' % (reward_sum, running_reward))

            reward_sum = 0
            observation = env.reset() # reset env
            nn_input, actions = get_reset_values(observation, nn, sess)
