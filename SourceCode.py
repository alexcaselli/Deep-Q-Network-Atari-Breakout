import os
import numpy as np
import matplotlib.pyplot as plt
os.environ.setdefault('PATH', '')
from collections import deque
import gym
from gym import spaces
import cv2
cv2.ocl.setUseOpenCL(False)
from gym.wrappers import TimeLimit
from collections import deque
import tensorflow as tf
from tensorflow.keras import utils
from tensorflow.python.ops.init_ops import VarianceScaling
from tqdm import tqdm
import time
from gym.wrappers import Monitor

# Directory to store data
directory = '/Desktop/directory'
if not os.path.exists(directory):
  os.makedirs(directory)


# Wrapper --------------------------

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1) #pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)

class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)

class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done  = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=np.uint8)
        self._skip       = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True, dict_space_key=None):
        """
        Warp frames to 84x84 as done in the Nature paper and later work.
        If the environment uses dictionary observations, `dict_space_key` can be specified which indicates which
        observation should be warped.
        """
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._key = dict_space_key
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
        )
        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, obs):
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]

        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self._width, self._height), interpolation=cv2.INTER_AREA
        )
        if self._grayscale:
            frame = np.expand_dims(frame, -1)

        if self._key is None:
            obs = frame
        else:
            obs = obs.copy()
            obs[self._key] = frame
        return obs


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[:-1] + (shp[-1] * k,)), dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))

class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0

class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        frames = self._force()
        return frames.shape[frames.ndim - 1]

    def frame(self, i):
        return self._force()[..., i]

def make_atari(env_id, max_episode_steps=None):
    env = gym.make(env_id)
    assert 'NoFrameskip' in env.spec.id
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env

def wrap_deepmind(env, episode_life=True, clip_rewards=True, frame_stack=False, scale=False):
    """Configure environment for DeepMind-style Atari.
    """
    if episode_life:
        env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 4)
    return env
    
def wrap_atari_deepmind(env_name, clip):
    atari = make_atari(env_name)
    wrapped = wrap_deepmind(atari, episode_life=True, clip_rewards=clip, frame_stack=True, scale=True)
    return wrapped

# End Wrapper -------------------------

# Replay Buffer -----------------------

class Replay:

  # sequence = ['s', 'a', 'r', 's_next', 'done']
  def __init__(self, N=10000):
        self.buffer = deque(maxlen=N)

  def add(self, transition):
        self.buffer.append(transition)

  # TODO: Possible refactor, np.array since the data declaration, we already know the leght (batch_size)
  def get_transitions(self, batch_size, with_replacement):
    available_trainsitions = range(len(self.buffer))
    transitions_idx = np.random.choice(available_trainsitions, size=batch_size, replace=with_replacement)
    batch = {'s': [], 'a': np.zeros((batch_size)), 'r': np.zeros((batch_size)), 's_next': [], 'terminal': np.ones((batch_size), dtype=bool)}
    i = 0
    for idx in transitions_idx:
      transition = self.buffer[idx]

      batch['s'].append(transition[0])
      batch['a'][i] = transition[1]
      batch['r'][i] = transition[2]
      batch['s_next'].append(transition[3])
      batch['terminal'][i] = transition[4]
      i += 1

    return batch

  @property
  def size(self):
    return len(self.buffer)


# End Replay buffer --------------------

# Compute the moving average
def return_per_episode(tot_reward, n):
    j = 0
    summe = 0
    means = []
    for i in range(len(tot_reward)):
        if not (i % n == 0):
            j = j + 1
            summe = summe + tot_reward[i]
        else:
            means.append([i,summe/n])
            j = 0
            summe = 0
    return means


# Class Agent --------------------------
class Agent:
    def __init__(self, C = 10_000, buff_size=10_000, batch_size=32, action_space=4, evaluate_each=100_000):

        self.lr = 1e-4
        self.decay = 0.99
        self.evaluate_each = evaluate_each
        self.n_actions = action_space
        self.n = 4

        # Create the 2 graphs
        self.graph_online = tf.Graph()
        with self.graph_online.as_default():
            self.online = self.create()

        self.graph_target = tf.Graph()
        with self.graph_target.as_default():
            self.target = self.create()

        self.buffer_size = buff_size
        self.batch_size = batch_size
        self.gamma = 0.99
        self.C = C
        self.update_counter = 0
        self.epsilon = 1.0
        self.min_epsilon = 0.1
        
        self.initialized = 0
        self.replay = Replay(N=buff_size)

        #Save score of each evaluation phase
        self.scores = []

        #Save losses of each train step
        self.losses = []

    
    

    def set_action_space_size(self, n_actions):
        self.n_actions = n_actions

    def save(self, saver, sess):
        print("Saving...")
        saver.save(sess, directory + '/online.ckpt') 
        print("Saved!")
    
    def copy(self, saver, sess):
        print("Restoring...")
        saver.restore(sess, directory + '/online.ckpt')
        print("Restored!")

    def create(self):

        X_img = tf.placeholder(tf.float32, shape=(None, 84, 84, 4))
        Y = tf.placeholder(tf.float32, shape=(None, ))
        a = tf.placeholder(tf.int32, shape=(None, 2 ))

        initializer = VarianceScaling(1.0)
        initializer_bias = tf.zeros_initializer()
        
        W_conv1 = tf.Variable(initializer([8, 8, 4, 32])) 
        b_conv1 = tf.Variable(initializer_bias(shape=(32,)))
        A_conv1 = tf.nn.relu(tf.nn.conv2d(X_img, W_conv1, strides=[1, 4, 4, 1], padding='SAME') + b_conv1)
    
        W_conv2 = tf.Variable(initializer([4, 4, 32, 64])) 
        b_conv2 = tf.Variable(initializer_bias(shape=(64,)))
        A_conv2 = tf.nn.relu(tf.nn.conv2d(A_conv1, W_conv2, strides=[1, 2, 2, 1],  padding='SAME') + b_conv2)

        W_conv3 = tf.Variable(initializer([3, 3, 64, 64])) 
        b_conv3 = tf.Variable(initializer_bias(shape=(64,)))
        A_conv3 = tf.nn.relu(tf.nn.conv2d(A_conv2, W_conv3, strides=[1, 1, 1, 1],  padding='SAME') + b_conv3)


        flatten = tf.reshape(A_conv3, [-1, 11*11*64]) 

    
        n_neurons_1 = 512
        n_neurons_2 = self.n_actions

        W_fc1 = tf.Variable(initializer([11*11*64, n_neurons_1]))
        b_fc1 = tf.Variable(initializer_bias(n_neurons_1))
        Z_fc1 = tf.nn.relu(tf.matmul(flatten, W_fc1) + b_fc1)
    

        W_fc2 = tf.Variable(initializer([n_neurons_1, n_neurons_2]))
        b_fc2 = tf.Variable(initializer_bias(n_neurons_2))
        Qs = tf.matmul(Z_fc1, W_fc2) + b_fc2

        action = tf.argmax(Qs, axis=1)

        max_q = tf.reduce_max(Qs, axis = 1)

    
        loss = tf.reduce_mean(tf.square(tf.stop_gradient(Y) - tf.gather_nd(Qs,a)))
        optimizer =  tf.train.RMSPropOptimizer(self.lr, self.decay)
        train = optimizer.minimize(loss)

        saver = tf.train.Saver()

        return (X_img, Y, train, loss, saver, W_conv2, action, Qs, a)


    def add_transition(self, transition):
        self.replay.add(transition)

    def compute_Qs(self, state):

        feed = {DQN_online[0]: state,
                DQN_online[1]: np.random.rand(len(state),  ),
                DQN_online[8]: np.ones((32,2), dtype=np.int32)}
        return sess_o.run(DQN_online[7], feed )

    def initialize(self):

        print( "initializing the networks...")
    
        self.save(DQN_online[4], sess_o)
        self.copy(DQN_target[4], sess_t)
    
        self.initialized = 1
        print( "Networks initialized!")

    def train(self, i):

        # If first train step
        if self.initialized == 0:
            self.initialize()

        if self.replay.size < self.buffer_size:
            return

    
        if i % self.n == 0:
            batch = self.replay.get_transitions(self.batch_size, False)

            states = batch['s']
        
            fake_targets = np.random.rand(len(states), )
            
            
            states_next = batch['s_next']
            feed_target = {DQN_target[0]: np.array(states_next), DQN_target[1]: fake_targets, DQN_target[8]: np.ones((32,2), dtype=np.int32)}
            Qs_batch_next = sess_t.run(DQN_target[7], feed_target)

            
            targets = []

            # Take the indexes of the actions
            num = np.arange(self.batch_size)
            idx = np.ones((self.batch_size,2), dtype=np.int32)
            idx[:,0] = num
            idx[:,1] = np.array(batch['a'])
            for index in range(len(batch['s'])):
                if not batch['terminal'][index]:
                    max_future_Q = np.max(Qs_batch_next[index])
                    Q_new = batch['r'][index] + self.gamma * max_future_Q
                else:
                    Q_new = batch['r'][index]

      
      
            targets.append(Q_new)
    

            # Train Online DQN
            
            feed_train = {DQN_online[0]: states, DQN_online[1]: targets, DQN_online[8]: idx}
            l, _ = sess_o.run([DQN_online[3], DQN_online[2]], feed_train)
            
            self.losses.append(l)

            if i % self.C == 0:
                print("Synchronizing the networks...")
                self.save(DQN_online[4], sess_o)
                self.copy(DQN_target[4], sess_t)


    # Take the e-greedy action
    def take_action(self, agent, env, state, e=1.0):
        if np.random.random() < 1-self.epsilon:
            a = np.argmax(agent.compute_Qs(np.array([state])))
        else:
            a = env.action_space.sample()  
        return a

    # Take action during the evaluation step
    def get_action(self, agent, env, state, e=0.001):
        if np.random.random() < 1-e:
            a = np.argmax(agent.compute_Qs(np.array([state])))
        else:
            a = env.action_space.sample()  
        return a   

# End of Agent -----------------------

# Main Training ----------------
def main(N=2_000_000):
    # Change this to 'AssaultNoFrameskip-v4' to train on the second game
    env = wrap_atari_deepmind('BreakoutNoFrameskip-v4', True)
    agent.set_action_space_size(env.action_space.n)

    # rewards of all the episodes
    tot_reward = []

    # Rewards of single episode
    rewards = []
    i = 0
    episode = 1
    pbar = tqdm(total = N)
    while i < N:
        r = 0
        s = env.reset()
        terminal = False
        episode_reward = 0
        while not terminal:
            a = agent.take_action(agent, env, np.array(s))
            s_next, r, terminal, dizi = env.step(a)
            episode_reward += r
            rewards.append(r)
            agent.add_transition([s, a, r, s_next, terminal])
            i = i+1
            if i % 10000 == 0:
                pbar.update(10000)
            
            agent.train(i)
            s = s_next

            if agent.epsilon > agent.min_epsilon:
                agent.epsilon -= 0.0000009
                agent.epsilon = max(agent.min_epsilon, agent.epsilon)

            if i % agent.evaluate_each == 0:
                evaluate()
          

            tot_reward.append(episode_reward)
            
        # uncomment to print episodes rewards
        #print('Episode: {0}. Reward: {1}.'.format(episode, episode_reward))
        episode = episode + 1
        

    pbar.close()

    # Save model
    print("Saving the last model..")
    agent.save(DQN_online[4], sess_o)

    # Save values
    print("Saving the data..")
    np.save(directory + "/episodes_rewards", np.array(tot_reward))
    np.save(directory + "/evaluation_scores", np.array(agent.scores))
    np.save(directory + "/training_loss", np.array(agent.losses))
      



    plt.plot(tot_reward, 'r')
    plt.show()
    print(agent.scores)
    plt.plot(agent.scores, 'r')
    plt.show()

    env.close()
# End Main -------------------

# Evaluation Phase
def evaluate():
    print("Evaluating...")
    # Change this to 'AssaultNoFrameskip-v4' to evaluate the second game
    env_e = wrap_atari_deepmind('BreakoutNoFrameskip-v4', False)
    tot_reward=[]
    episode = 1
    i=0
    play_scores = []
    for play in range(30):
        summ_play = 0
        for ep in range(5):
            r = 0
            s = env_e.reset()
            terminal = False
            episode_reward = 0
            while not terminal:
                a = agent.get_action(agent, env_e, np.array(s))
                s_next, r, terminal, dizi = env_e.step(a)
                episode_reward += r
                i = i+1
                s = s_next
            summ_play = summ_play + episode_reward
            tot_reward.append(episode_reward)
            episode = episode + 1
        play_scores.append(summ_play)
        print("Play {0} score: {1}".format(play, summ_play))
    play_mean = np.mean(np.array(play_scores))
    print("Average: ", play_mean)
    agent.scores.append(play_mean)
    print("Stop Evaluation...")
    env_e.close()


# Play for 1000 steps
def play(N=1000):

    # Change this to 'AssaultNoFrameskip-v4' to play the second game
    env = wrap_atari_deepmind('BreakoutNoFrameskip-v4', False)
    env = Monitor(env, directory + "/", force=True)
    agent.copy(DQN_online[4], sess_o)
    tot_reward=[]
    episode = 1
    i=0
    while i < N:
        
        r = 0
        s = env.reset()
        terminal = False
        episode_reward = 0
        while not terminal:
          
            env.render()
            a = agent.get_action(agent, env, np.array(s))
            s_next, r, terminal, dizi = env.step(a)
            episode_reward += r
            i = i+1
            s = s_next
        tot_reward.append(episode_reward)
        print("Episode reward: ", episode_reward)
        episode = episode + 1
    env.close()









# Run the code
print("Creating sessions...")

agent = Agent(C = 10_000, buff_size=10_000, batch_size=32, action_space=4, evaluate_each=100_000)

with agent.graph_online.as_default():
  DQN_online = agent.online

with agent.graph_target.as_default():
  DQN_target = agent.target

with tf.Session(graph=agent.graph_online) as sess_o:
  sess_o.run(tf.global_variables_initializer())

  with tf.Session(graph=agent.graph_target) as sess_t:
    sess_t.run(tf.global_variables_initializer())

    # Train
    main()

    #Play 
    play()
  
