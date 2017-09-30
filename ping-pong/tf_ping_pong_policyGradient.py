""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
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
#learning_rate = 1e-4
learning_rate = 1e-3
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
render = False

save_freq = 100 # keep zero if you dun want to save model
plot_freq = 100 # keep zero if you dun want to draw the scores

model_save_path = os.path.join(os.getcwd(),'model_tf_policyGrad','mymodel.ckpt')

# model initialization
D = 80 * 80 # input dimensionality: 80x80 grid

# replaces model initializations, forward_prop and back_prop
class neural_network():
    def __init__(self, D, H, learning_rate):
        self.input_layer = tf.placeholder(shape=[None,D],dtype=tf.float32)
        self.hidden_layer = slim.fully_connected(self.input_layer, H, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), biases_initializer=None)
        self.output_layer = slim.fully_connected(self.hidden_layer, 1, activation_fn=tf.nn.sigmoid, weights_initializer=tf.contrib.layers.xavier_initializer() ,biases_initializer=None)

        self.actual_actions = tf.placeholder(shape=[None],dtype=tf.int32)
        self.game_rewards = tf.placeholder(shape=[None,1],dtype=tf.float32)
        
        comparison = tf.equal(self.actual_actions,tf.constant(1))
        
        # loss_fn = Ya(log(Yp)) + (1-Ya)log(1-Yp)
        self.log_loss = tf.where(comparison, tf.log(self.output_layer), tf.log(tf.subtract(1.0,self.output_layer)))
        self.policy_loss = -tf.reduce_mean(self.log_loss * self.game_rewards)
        
        w_variables = tf.trainable_variables()
        self.gradients = []
        for indx,w in enumerate(w_variables):
            w_holder_var = tf.placeholder(tf.float32,name="w_"+ str(indx))
            self.gradients.append(w_holder_var)
        
        self.all_gradients = tf.gradients(self.policy_loss,w_variables)
        #optimizer = tf.train.RMSPropOptimizer(decay = decay_rate, learning_rate=learning_rate)
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


# discount_rewards(np.array([1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,1.0]))
# returns: array([ 1., 0.96059601, 0.970299, 0.9801, 0.99, 1., 0.9801, 0.99, 1.])
def discount_rewards(r):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(range(0, r.size)):
    if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r


env = gym.make("Pong-v0")
observation = env.reset()
prev_x = None # used in computing the difference frame
xs,hs,dlogps,drs,all_game_scores = [],[],[],[],[]
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
        aprob, h = sess.run([nn.output_layer, nn.hidden_layer], feed_dict={nn.input_layer: np.reshape(x,(1,x.shape[0]))})
        # action = 2 is 1
        # action = 3 is 0
        action = 2 if np.random.uniform() < aprob else 3 # roll the dice!
        
        # record various intermediates (needed later for backprop)
        xs.append(x) # observation
        hs.append(h) # hidden state
        y = 1 if action == 2 else 0 # a "fake label"
        dlogps.append(y) # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)
        
        # step the environment and get new measurements
        observation, reward, done, info = env.step(action)
        reward_sum += reward
        drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)

        if done: # an episode finished
            episode_number += 1
            # stack together all inputs, hidden states, action gradients, and rewards for this episode
            epx = np.vstack(xs)
            eph = np.vstack(hs)
            epdlogp = np.vstack(dlogps)
            epr = np.vstack(drs)
            xs,hs,dlogps,drs = [],[],[],[] # reset array memory
            # compute the discounted reward backwards through time
            discounted_epr = discount_rewards(epr)
            # standardize the rewards to be unit normal (helps control the gradient estimator variance)
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)

            grads = sess.run(nn.all_gradients, feed_dict = {nn.input_layer: epx, nn.game_rewards:discounted_epr, nn.actual_actions: epdlogp.ravel()})
            for indx,grad in enumerate(grads):
                grad_buffer[indx] += grad
                
            # perform rmsprop parameter update every batch_size episodes
            if episode_number % batch_size == 0:
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
                plt.title('Tensorflow policy gradient') 
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
