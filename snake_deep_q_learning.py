import tensorflow as tf
from ple import PLE
import ple.games as game
import random as random
import collections as col
import numpy as num
import itertools
import sys
import queue

# Tensorflow functions :

# Set random values from a truncated normal distribution
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

# good practice to initialize a slightly positive bias to avoid "dead neurons"
def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


def relu(input, w, st, b):
    return tf.nn.relu(tf.nn.conv2d(input, w, strides=st, padding="SAME") + b)

# Max pooling
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def matmul(hidden_conv_flat, w_feed_forward, b_feed_forward):
    return tf.matmul(hidden_conv_flat, w_feed_forward) + b_feed_forward


# ///////////////////////////////////////////////////////////////////////

def init_neural_network(screen_size_x, screen_size_y, nb_actions, nb_frames=4):
    # First Convolutional Layer
    # 32 Features for each 8x8 patch, input channels = 4
    # bias vector with a component for each output channel
    W_conv1 = weight_variable([8, 8, nb_frames, 32])
    b_conv1 = bias_variable([32])
    # Second Convolutional Layer
    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])
    # Third Convolutional Layer
    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])

    W_feed_forward1 = weight_variable([256, 256])
    b_feed_forward1 = bias_variable([256])
    W_feed_forward2 = weight_variable([256, nb_actions])
    b_feed_forward2 = bias_variable([nb_actions])

    # init input layer
    input_l = tf.placeholder("float", [None, screen_size_x, screen_size_y, nb_frames])

    hidden_conv_l1 = relu(input_l, W_conv1, [1, 4, 4, 1], b_conv1)
    hidden_max_pooling_l1 = max_pool_2x2(hidden_conv_l1)
    hidden_conv_l2 = relu(hidden_max_pooling_l1, W_conv2, [1, 2, 2, 1], b_conv2)
    hidden_max_pooling_l2 = max_pool_2x2(hidden_conv_l2)
    hidden_conv_l3 = relu(hidden_max_pooling_l2, W_conv3, [1, 1, 1, 1], b_conv3)
    hidden_max_pooling_l3 = max_pool_2x2(hidden_conv_l3)


    hidden_conv_l3_flat = tf.reshape(hidden_max_pooling_l3, [-1, 256])

    final_hidden_activations = tf.nn.relu(matmul(hidden_conv_l3_flat, W_feed_forward1, b_feed_forward1))
    output_l = matmul(final_hidden_activations, W_feed_forward2, b_feed_forward2)
    return input_l, output_l

class MyAgent(object):
    def __init__(self, actionset, wantDebug):
        self.actionset = actionset
        self.wantDebug = wantDebug
        if self.wantDebug:
            for i in range(0, len(actionset)):
                print("Action number: " + str(i) + " " + str(actionset[i]))

    def pickRandomAction(self):
        # actions -> "up": K_w, "left": K_a, "right": K_d, "down": K_s, "none": -
        return self.actionset[random.randint(0, 4)]

    def test_random(self, p, snake, nb_frames_test):
        scores = []
        p.init()
        for f in range(nb_frames_test):
            if p.game_over():
                scores.append(snake.getScore() + 5)  # save the scores
                p.reset_game()
            else:
                max_action = self.pickRandomAction()  # take a random action
                p.act(max_action)  # execute action
            if nb_frames_test == f and not p.game_over():
                scores.append(snake.getScore())
        return scores

    def test(self, p, snake, nb_frames_test,session,input_l,output_l):
        scores = []
        p.init()
        current_state = p.getScreenGrayscale()
        for f in range(nb_frames_test):
            if p.game_over():
                scores.append(snake.getScore() + 5)  # save the scores
                p.reset_game()
            else:
                action_index=num.max(session.run(output_l,feed_dict={input_l:[current_state]}))
                p.act(self.actionset[action_index])  # execute action
            current_state = p.getScreenGrayscale()

        return scores

    def getSmartReward(self, current_state_snake, next_state_snake, reward):
        if reward == -5:  # game over
            return -5
        elif reward == 1:  # Snake found apple
            return 4
        else:
            if self.foodIsNearer(current_state_snake, next_state_snake):  # snake goes to apple
                return 1
            return -1  # snake goes away from apple

    def foodIsNearer(self, state1, state2):
        if (abs(state1["snake_head_x"] - state1["food_x"]) + abs(state1["snake_head_y"] - state1["food_y"])
                > abs(state2["snake_head_x"] - state2["food_x"]) + abs(state2["snake_head_y"] - state2["food_y"])):
            if self.wantDebug:
                print("Movement in right direction")
            return True
        else:
            if self.wantDebug:
                print("Movement in wrong direction")
            return False

    def generate_max_action(self):
        pass
    # deep Q-Leaning Algorithm
    # returns a trained neuronal network
    def train(self, p, snake, nb_frames, gamma, learner, explored, screen_size_x=64, screen_size_y=64):
        # init game
        p.init()
        nb_actions = 4  # number of possible actions
        input_frames = 4  # number of frames to consider as input
        session = tf.Session()
        # Create neuronal network with 3 layers
        input_l, output_l = init_neural_network(screen_size_x, screen_size_y, input_frames, nb_actions)
        action = tf.placeholder("float", [None, nb_actions])
        target = tf.placeholder("float", [None])
        readout_action = tf.reduce_sum(tf.multiply(output_l, action), reduction_indices=1)
        cost = tf.reduce_mean(tf.square(target - readout_action))
        train_operation = tf.train.AdamOptimizer(1e-6).minimize(cost)
        session.run(tf.global_variables_initializer())

        #///////////////////////////////////////////////////////////
        # init replay memory as set
        replay_memory = col.deque(maxlen=1000)#queue.Queue(maxsize=1000)
        current_state_snake = snake.getGameState()

        xt= p.getScreenGrayscale()#TODO format screen
        for t in range(1,nb_frames):
            if p.game_over():
                p.reset_game()
                xt = p.getScreenGrayscale()
                current_state_snake = snake.getGameState()

            print(xt)
            if random.random() <= explored:
                action_index = num.max(session.run(output_l,feed_dict={input_l:[xt]}))
            else:
                action_index = random.randint(0,4)

            reward = p.act(self.actionset[action_index])
            next_state_snake= snake.getGameState()
            xt1 = p.getScreenGrayscale()#TODO format screen

            smart_reward= self.getSmartReward(current_state_snake, next_state_snake, reward)
            if len(replay_memory)==replay_memory.maxlen:
                replay_memory.get()
            replay_memory.append((xt,action_index,smart_reward,xt1))


            # sample random minibatch of transitions from replay memory
            mini_batch_size=4
            if len(replay_memory)>mini_batch_size:
                mini_batch = random.sample(list(replay_memory), mini_batch_size)


                # mini_batch variables:
                yj = [] # expected rewards
                prev_states = [d[0] for d in list(mini_batch)]
                actions = [d[1] for d in list(mini_batch)]
                curr_states = [d[3] for d in list(mini_batch)]
                reward_action = output_l.eval(feed_dict={input_l: curr_states},session=session)
                for j in range(0,mini_batch_size):
                    sequence=mini_batch[j]
                    if sequence[2]==-5:# if reward is - 5 next state is terminal state
                        yj.append(sequence[2])
                    else:
                        yj.append(sequence[2]+gamma*num.max(reward_action[j]))
                # gradient descent step
                train_operation.run(feed_dict={input_l: prev_states,action: actions,target: yj})
            current_state_snake=next_state_snake
            xt=xt1
        return session,input_l,output_l


def main():
    snake = game.Snake(width=64, height=64)

    p = PLE(snake, fps=10, display_screen=True)

    myAgent = MyAgent(p.getActionSet(), False)

    '''
    input = sys.argv[1]
    gamma = float(input[0]) / 10
    learner = float(input[1]) / 10
    explored = float(input[2]) / 10
    '''
    session, input_l, output_l = myAgent.train(p, snake, 100, 0.1, 0.1, 0.1)

    learned_scores = myAgent.test( p, snake, 100,session, input_l, output_l)
    result = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    for elem in learned_scores:
        result[int(elem)] += 1
    for i in range(0,len(result)-1):
        print(result[i],end=";")
    print(result[len(result)-1])


if __name__ == '__main__': main()
