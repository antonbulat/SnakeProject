import tensorflow as tf
from ple import PLE
import ple.games as game
import numpy.random as num
import collections as col
import sys

# Tensorflow functions :
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)

def relu(input,w,st,b):
    return tf.nn.relu(tf.nn.conv2d(input, w, strides=st, padding="SAME") + b)

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

def matmul(hidden_conv_flat,w_feed_forward,b_feed_forward):
    return tf.matmul(hidden_conv_flat, w_feed_forward) + b_feed_forward


#///////////////////////////////////////////////////////////////////////

def init_neural_network(screen_size_x,screen_size_y,nb_frames,nb_actions):
    # First Convolutional Layer
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

    hidden_conv_l1 = relu(input_l,W_conv1,[1, 4, 4, 1],b_conv1)
    hidden_max_pooling_l1 = max_pool_2x2(hidden_conv_l1)
    hidden_conv_l2 =relu(hidden_max_pooling_l1,W_conv2,[1, 2, 2, 1],b_conv2)
    hidden_max_pooling_l2 = max_pool_2x2(hidden_conv_l2)
    hidden_conv_l3 = relu(hidden_max_pooling_l2,W_conv3,[1, 1, 1, 1],b_conv3)
    hidden_max_pooling_l3 = max_pool_2x2(hidden_conv_l3)

    hidden_conv_l3_flat = tf.reshape(hidden_max_pooling_l3, [-1, 256])

    final_hidden_activations = tf.nn.relu(matmul(hidden_conv_l3_flat,W_feed_forward1,b_feed_forward1))
    output_l = matmul(final_hidden_activations, W_feed_forward2,b_feed_forward2)
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
        return self.actionset[num.randint(0, 4)]

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
    #TODO change to use euronal network
    def test(self, q_dic, p, snake, nb_frames_test):
        scores = []
        p.init()
        temp_state = snake.getGameState()
        current_state = hash(str(temp_state))
        for f in range(nb_frames_test):
            if p.game_over():
                scores.append(snake.getScore() + 5)  # save the scores
                p.reset_game()
            else:
                max_action = self.pickRandomAction()  # first take a random action
                max = 0
                if (current_state, max_action) in q_dic:
                    max = q_dic[(current_state, max_action)]

                # look if there is an other action in actionset which is better (based on q table)
                for a in self.actionset:
                    if (current_state, a) in q_dic:
                        tmp = q_dic[(current_state, a)]
                        if tmp > max:
                            max = tmp
                            max_action = a

                p.act(max_action)  # execute action

            temp_state = snake.getGameState()
            current_state = hash(str(temp_state))

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

    # deep Q-Leaning Algorithm
    # returns a trained neuronal network
    # TODO... everything
    # IDEAS from https://gist.github.com/DanielSlater/f611a3aa737d894b689f#file-gistfile1-txt
    # and from https://www.tensorflow.org/versions/r0.12/tutorials/mnist/pros/
    # TODO delete previous line
    def train(self, p, snake, nb_frames, gamma, learner, explored, dic,screen_size_x=64,screen_size_y=64):
        #init game
        p.init()
        nb_actions=4# number of possible actions
        input_frames=4 # number of frames to consider as input
        # init replay memory as set
        replay_memory = {}
        previous_observations = col.deque()
        session = tf.Session()
        # Create neuronal network with 3 layers
        input_l, output_l=init_neural_network(screen_size_x, screen_size_y, input_frames, nb_actions)
        action = tf.placeholder("float", [None, nb_actions])
        target = tf.placeholder("float", [None])
        readout_action = tf.reduce_sum(tf.multiply(output_l, action), reduction_indices=1)
        cost = tf.reduce_mean(tf.square(target - readout_action))
        train_operation = tf.train.AdamOptimizer(1e-6).minimize(cost)
        session.run(tf.initialize_all_variables())
'''
        if num.random() > explored:
            action = max_action

        else:
            action = self.pickRandomAction()
'''

def main():
    snake = game.Snake(width=64, height=64)

    p = PLE(snake, fps=10, display_screen=False)

    myAgent = MyAgent(p.getActionSet(), False)

    dic = {}
    input = sys.argv[1]
    gamma = float(input[0]) / 10
    learner = float(input[1]) / 10
    explored = float(input[2]) / 10

    my_q_dic = myAgent.train(p, snake, 100000, gamma, learner, explored, dic)

    learned_scores = myAgent.test(my_q_dic, p, snake, 100000)
    title = "Configuration: gamma: " + str(gamma) + " ,learning_rate: " \
            + str(learner) + " ,explored: " + str(explored) + " \nNum of train rounds: 100000"
    print("----------------------------------------------------")
    print(title)
    print(learned_scores)
    result=[0,0,0,0,0,0,0,0,0]
    for elem in learned_scores:
        result[int(elem)]+=1
    print("Scores of the training: ")
    print(result)

if __name__ == '__main__': main()