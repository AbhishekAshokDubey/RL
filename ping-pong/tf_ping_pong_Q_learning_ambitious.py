""" Trains an agent with (stochastic) Q Learning on Pong. Uses OpenAI Gym. """
import numpy as np
#import cPickle as pickle
import gym
import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
import matplotlib.pyplot as plt

# hyperparameters
H = 200 # number of hidden layer neurons
batch_size = 10 # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
randomness = 0.1 # initial prob to corrupt (randomness) the action
render = False

save_freq = 100 # keep zero if you dun want to save model
plot_freq = 100 # keep zero if you dun want to draw the scores

model_save_path = os.path.join(os.getcwd(),'model_QLearning','mymodel.ckpt')

# model initialization
D = 80 * 80 # input dimensionality: 80x80 grid

# replaces model initializations, forward_prop and back_prop
class neural_network():
    def __init__(self, D, H, learning_rate):        
        self.input_layer = tf.placeholder(shape=[None,D],dtype=tf.float32)
        self.hidden_layer = slim.fully_connected(self.input_layer, H, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), biases_initializer=None)
        self.output_layer = slim.fully_connected(self.hidden_layer, 2, activation_fn=tf.nn.sigmoid, weights_initializer=tf.contrib.layers.xavier_initializer() ,biases_initializer=None)
        
        self.max_rew_action = tf.argmax(self.output_layer,1)
        
        self.updated_Q_vaules = tf.placeholder(shape=[None,2],dtype=tf.float32)
        self.loss = tf.reduce_sum(tf.square(self.updated_Q_vaules - self.output_layer))

        w_variables = tf.trainable_variables()
        self.gradients = []
        for indx,w in enumerate(w_variables):
            w_holder_var = tf.placeholder(tf.float32,name="w_"+ str(indx))
            self.gradients.append(w_holder_var)
        
        self.all_gradients = tf.gradients(self.loss,w_variables)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.apply_grads = optimizer.apply_gradients(zip(self.gradients,w_variables))
        
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
  return I.astype(np.float).ravel() # convert to 1D array and return


env = gym.make("Pong-v0")
observation = env.reset()
prev_x = None # used in computing the difference frame
xs,hs,dlogps,drs,qs,all_game_scores = [],[],[],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 0

tf.reset_default_graph()
nn = neural_network(D, H, learning_rate)

with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(model_save_path))
    if ckpt and ckpt.model_checkpoint_path:
        print("using the saved model")
        nn.saver.restore(sess, model_save_path)
    else:
        sess.run(tf.global_variables_initializer())
        print("making new model")
        
    grad_buffer = sess.run(tf.trainable_variables())
    for indx,grad in enumerate(grad_buffer):
        grad_buffer[indx] = grad * 0
    while True:
        if render: env.render()
        # preprocess the observation, set input to network to be difference image
        cur_x = prepro(observation)
        x = cur_x - prev_x if prev_x is not None else np.zeros(D)
        prev_x = cur_x
        # forward the policy network and sample an action from the returned probability
        y, q_vaules = sess.run([nn.max_rew_action, nn.output_layer], feed_dict={nn.input_layer: np.reshape(x,(1,x.shape[0]))})
        y = y[0]
        # action = 2 is 1
        # action = 3 is 0
        if np.random.uniform() < randomness: y = abs(1-y) # roll the dice!
        # record various intermediates (needed later for backprop)
        xs.append(x) # observation
        
        action = 2 if y else 3
        dlogps.append(y) # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)
        
        # step the environment and get new measurements
        observation, reward, done, info = env.step(action)
        
        # reduntant code, but keeping it for clarity of the concepts
        cur_x_temp = prepro(observation)
        x_next_temp = cur_x_temp - prev_x

        reward_sum += reward
        drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)

        next_state_q_vaules = sess.run(nn.output_layer, feed_dict={nn.input_layer: np.reshape(x_next_temp,(1,x_next_temp.shape[0]))})
        max_q_val = np.max(next_state_q_vaules)
        updated_q_values = q_vaules
        updated_q_values[0,y] = reward + (gamma * max_q_val) # bellman equation
#        print(q_vaules,updated_q_values)
        qs.append(updated_q_values)
                
        if done: # an episode finished
            episode_number += 1
            # stack together all inputs, hidden states, action gradients, and rewards for this episode
            epx = np.vstack(xs)
            epq = np.vstack(qs)
            epr = np.vstack(drs)
            xs,qs,drs = [],[],[] # reset array memory
            
            grads = sess.run(nn.all_gradients, feed_dict={nn.input_layer: epx, nn.updated_Q_vaules:epq})
            for indx,grad in enumerate(grads):
                grad_buffer[indx] += grad
                
            # perform rmsprop parameter update every batch_size episodes
            if episode_number % batch_size == 0:
                # reducing the randomness, after each batch
                randomness = randomness / (((episode_number*1.0) / 100) + 10)
                print("updating weights of the network")
                feed_dict = dict(zip(nn.gradients, grad_buffer))
                _ = sess.run(nn.apply_grads, feed_dict=feed_dict)
                for indx,grad in enumerate(grad_buffer):
                    grad_buffer[indx] = grad * 0

            if (save_freq and not(episode_number%save_freq)):
                print("saving the model ...")
                nn.saver.save(sess, model_save_path)
            if (plot_freq and not(episode_number%plot_freq)):
                #Tools > preferences > IPython console > Graphics > Graphics backend > Backend: Automatic
                #Then close and open Spyder.
                plt.clf()
                plt.plot(all_game_scores)
                plt.title('Tensorflow Q learning') 
                plt.pause(0.0001)

            # boring book-keeping
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            print('resetting env. episode reward %f. running mean: %f' % (reward_sum, running_reward))
            #        if episode_number % 100 == 0: pickle.dump(model, open('save.p', 'wb'))
            all_game_scores.append(reward_sum+21.0)
            reward_sum = 0
            observation = env.reset() # reset env
            prev_x = None
#        if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
#            print(('ep %d: game finished, reward: %f' % (episode_number, reward)) + ('' if reward == -1 else ' !!!!!!!!'))
