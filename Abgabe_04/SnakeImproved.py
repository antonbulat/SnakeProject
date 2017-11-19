import ple.games as game
import numpy.random as num
import pickle as pic
import os
import psutil
import time
import matplotlib.pyplot as plt
from ple import PLE


# naive hash function for the screen
def my_hash(screen):
    return hash(str(screen))


class MyAgent(object):
    def __init__(self, actionset):
        self.actionset = actionset

    def pickRandomAction(self):
        # actions -> "up": K_w, "left": K_a, "right": K_d, "down": K_s, "none": -
        return self.actionset[num.randint(0, 4)]

    # TODO: diese Funktion muss noch angepasst werden auf die Dictionary Struktur
    def pickBestAction(self, state, dict):
        max = 0
        counter = 0
        max_c = 0
        for v in dict[state]:
            if (max > v):
                max = v
                max_c = counter
            counter += 1
        return dict[state].__getitem__(max_c)

    def test_random(self, p, snake, nb_frames):
        scores = []
        p.init()
        for f in range(nb_frames):
            if p.game_over():
                scores.append(snake.getScore() + 5)  # save the scores
                p.reset_game()
            else:
                max_action = self.pickRandomAction()  # take a random action
                p.act(max_action)  # execute action
        return scores

    def test(self, q_dic, p, snake, nb_frames):
        scores = []
        p.init()
        current_state = my_hash(p.getScreenRGB())  # make the screen hashable
        for f in range(nb_frames):
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
                current_state = my_hash(p.getScreenRGB())  # get next state

        return scores

    # deleted bug... now you can play with diffrent rewards

    def getSmartReward(self, current_state, next_state, reward):
        if reward == -5:  # game over
            return -5
        elif reward == 1:  # Snake found apple
            return 3
        else:
            if self.foodIsNearer(current_state, next_state):  # snake goes to apple
                return 0.5
            return 0  # snake goes away from apple

    def foodIsNearer(self, state1, state2):
        if (abs(state1["snake_head_x"] - state1["food_x"]) + abs(state1["snake_head_y"] - state1["food_y"])
                > abs(state2["snake_head_x"] - state2["food_x"]) + abs(state2["snake_head_y"] - state2["food_y"])):
            return True
        else:
            return False

    # Q-Leaning Algorithm
    # returns a q-dictionary
    def train(self, p, snake, nb_frames, gamma, learn, explore, q_dic={}):
        p.init()
        # because of the number of states we work with a dictionary with (state,action) as key
        # so we always have to check if the key is already in the dictionary
        # - if not we have to initialize the key with 0

        # and newly proposed Q-value is taken into account
        current_state = my_hash(p.getScreenRGB().tolist())  # make the screen hashable
        current_state_snake = snake.getGameState()
        action = self.pickRandomAction()  # select a random action
        for f in range(nb_frames):
            if p.game_over():
                p.reset_game()
            # //////////////////////////////////////////////////////////////////////////////////////////
            else:
                reward = p.act(action)  # get reward
                next_state_snake = snake.getGameState()
                smartReward = self.getSmartReward(current_state_snake, next_state_snake, reward)
                next_state = my_hash(p.getScreenRGB().tolist())  # get next state
                if (current_state, action) not in q_dic:  # if current state is not in the dictionary
                    q_dic[(current_state, action)] = 0.0  # add initial value

                # find action with maximal q value:

                max_action = self.pickRandomAction()  # first take a random action
                max = 0
                if (next_state, max_action) in q_dic:
                    max = q_dic[(next_state, max_action)]
                else:
                    q_dic[(next_state, max_action)] = 0.0
                # look if the is an other action in actionset which is better (based on q table)
                for a in self.actionset:
                    if (next_state, a) in q_dic:
                        tmp = q_dic[(next_state, a)]
                        if tmp > max:
                            max = tmp
                            max_action = a
                    else:
                        q_dic[(next_state, a)] = 0.0
                # We are using SmartReward now. This means we use the information of the snake and food position.
                q_dic[(current_state, action)] += learn * (
                smartReward + gamma * q_dic[(next_state, max_action)] - q_dic[(current_state, action)])
                current_state = next_state
                current_state_snake = next_state_snake
                if num.random() > explore:
                    action = max_action
                else:
                    action = self.pickRandomAction()
        return q_dic


snake = game.Snake()

p = PLE(snake, fps=30, display_screen=False)

myAgent = MyAgent(p.getActionSet())
# p.init()
# start_time = time.time()
# my_q_dic = myAgent.train(p, snake,100000,0.5,0.7,0.8)
# end_time = time.time()

# print("TIME OF Q_LEARNING", end_time - start_time)

# try to get the memory info of the current process
# process = psutil.Process(os.getpid())
# print("PROZESS MEMORY INFO: ", process.memory_info().rss)

# save the q-dic in a file
# afile = open(os.getcwd() + '\q_dic.pkl', 'wb')
# pic.dump(my_q_dic, afile)
# afile.close()


# learned_scores = myAgent.test(my_q_dic, p, snake,100000)

# random_scores = myAgent.test_random(p, snake,100000)
# plt.ylabel("Number of games")
# plt.xlabel("Number of apples")
# plt.hist(random_scores, color='red')
# plt.hist(learned_scores, color='green')
# plt.show()


# Test diffrent configurations of gamma, explored, lerner...
# Show in Histogram
# Later take best configuration and learn more rounds
# //////////////////////////////////////////////////////////////////////////////////////////////

'''
test_scores=[]
configurations=[0.1,0.4,0.8]
i=0
for gamma in configurations:
    for lerner in configurations:
        for explored in configurations:
            my_q_dic = myAgent.train(p, snake, 100000, 0.5, 0.7, 0.8)
            afile = open(os.getcwd() + '\q_dicG'+str(int(gamma*10))
                         +'L'+str(int(lerner*10))+'E'+str(int(explored*10))+'.pkl', 'wb')
            pic.dump(my_q_dic, afile)
            afile.close()
            learned_scores = myAgent.test(my_q_dic, p, snake, 100000)
            test_scores.append(learned_scores)
            print(i)
            i+=1

i=0
for gamma in configurations:
    for lerner in configurations:
        for explored in configurations:
            title = "Configuration: gamma: "+str(gamma)+" ,learning_rate: "+str(lerner)+" ,explored: "+str(explored)
            plt.title(title)
            plt.ylabel("Number of games")
            plt.xlabel("Number of apples")
            plt.hist(test_scores[i], color='green')
            plt.show()
            i+=1
'''''
gamma = 0.5
learner = 0.7
explored = 0.4

out_file = open(os.getcwd() + '\q_dic600000.pkl', 'wb')
infile = open(os.getcwd() + '\q_dic500000.pkl', 'rb')
dic = pic.load(infile)

my_q_dic = myAgent.train(p, snake, 100000, gamma, learner, explored, dic)

pic.dump(my_q_dic, out_file)
out_file.close()
infile.close()
learned_scores = myAgent.test(my_q_dic, p, snake, 100000)

title = "Configuration: gamma: " + str(gamma) + " ,learning_rate: " \
        + str(learner) + " ,explored: " + str(explored) + " Num of train rounds: 600000"
plt.title(title)
plt.ylabel("Number of games")
plt.xlabel("Number of apples")
plt.hist(learned_scores, color='green')
plt.show()
