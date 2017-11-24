import ple.games as game
import numpy.random as num
import pickle as pic
import os
#import psutil
import time
import matplotlib.pyplot as plt
from ple import PLE

# naive hash function for the screen
def my_hash(screen):
    return hash(str(screen))


class MyAgent(object):
    def __init__(self, actionset):
        self.actionset = actionset
        self.wantDebug = True

    def pickRandomAction(self):
        #actions -> "up": K_w, "left": K_a, "right": K_d, "down": K_s, "none": -
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
        scores.append(snake.getScore())
        return scores

    def test(self, q_dic, p, snake, nb_frames_test):
        scores = []
        p.init()
        temp_state = snake.getGameState()
        temp_state["food_x"] = 0
        temp_state["food_y"] = 0
        current_state = str(temp_state)
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
                temp_state["food_x"] = 0
                temp_state["food_y"] = 0
                current_state = str(temp_state)

        return scores

    def getSmartReward(self, current_state, next_state, reward):
        if reward == -5:  # game over
            return -5
        elif reward == 1:  # Snake found apple
            return 4
        else:
            if self.foodIsNearer(current_state, next_state):  # snake goes to apple
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

    # Q-Leaning Algorithm
    # returns a q-dictionary
    def train(self, p, snake, nb_frames, gamma, learner, explored, dic):
        p.init()

        #Counter
        state_add_count=0
        state_update_count = 0
        games_played_count = 1
        zeros_in_q_dict = 0
        not_zeros_in_q_dict = 0

        # because of the number of states we work with a dictionary with (state,action) as key
        # so we always have to check if the key is already in the dictionary
        # - if not we have to initialize the key with 0
        q_dic = dic
        #gamma = 0.75
        #learn = 1  # discount factor

        current_state_snake = snake.getGameState()
        current_state_snake_manipulated = snake.getGameState()
        current_state_snake_manipulated["food_x"]=0
        current_state_snake_manipulated["food_y"] = 0
        current_state = str(current_state_snake_manipulated)

        if self.wantDebug:
            print("Start state: " + str(current_state_snake))
            print("Start state no food: " + str(current_state_snake_manipulated))
            print("Start state hash: " + str(current_state))

        action = self.pickRandomAction()  # select a random action
        for f in range(nb_frames):
            if p.game_over():
                p.reset_game()
                games_played_count += 1
                if self.wantDebug:
                    print("Game over! Reset Game")
            # //////////////////////////////////////////////////////////////////////////////////////////
            else:
                if self.wantDebug:
                    print("-------ITERATION " + str(f) +"-------")

                reward = p.act(action)  # one movement

                #For reward value calculation we use the positioning of the food. But not for the Q table. We want to reduce the number of states
                next_state_snake = snake.getGameState()
                next_state_snake_manipulated = snake.getGameState()
                next_state_snake_manipulated["food_x"] = 0
                next_state_snake_manipulated["food_y"] = 0
                next_state = str(next_state_snake_manipulated)
                smartReward = self.getSmartReward( current_state_snake, next_state_snake, reward)

                if self.wantDebug:
                    print("Current state hash: " + str(current_state))
                    print("Next state hash: " + str(next_state))
                    print("Action: " + str(action))
                    print("Reward: " + str(smartReward))

                if ((current_state, action) not in q_dic):  # if current state is not in the dictionary
                    state_add_count += 1
                    q_dic[(current_state, action)] = 0.0  # add initial value
                else:
                    if self.wantDebug:
                        print("Aktual value of the current state and action: " + str(q_dic[(current_state, action)]))

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
                    #else:
                        #q_dic[(next_state, a)] = 0.0

                #We are using SmartReward. This means we use the information of the snake and food position for the reward function.
                tempvar = q_dic[(current_state, action)]
                q_dic[(current_state, action)] += learner * (smartReward + gamma * q_dic[(next_state, max_action)]- q_dic[(current_state, action)])

                if tempvar != q_dic[(current_state, action)]:
                    state_update_count += 1

                if self.wantDebug:
                    print("New value of the current state and action: " + str(q_dic[(current_state, action)]))

                current_state = next_state
                current_state_snake = next_state_snake

                if num.random() > explored:
                    action = max_action
                    if self.wantDebug:
                        print("Max action movement")
                else:
                    action = self.pickRandomAction()
                    if self.wantDebug:
                        print("Random movement")

                if self.wantDebug:
                    print("New action: " + str(action))
                #print("States added: " + str(state_add_count))
                #print("States updated: " + str(state_update_count))
                #time.sleep(1)

        if self.wantDebug:
            for val in q_dic:
                #print(str(val))
                #print(">>>" + str(q_dic[val]))
                if q_dic[val]==0:
                    zeros_in_q_dict += 1
                else:
                    not_zeros_in_q_dict +=1

            print("-------Finish learning-------")
            print("States added: " + str(state_add_count))
            print("States updated: " + str(state_update_count))
            print("Games played: " + str(games_played_count))
            print("Q Dictionary entries: " + str(q_dic.__len__()))
            print("Zeros in Q: " + str(zeros_in_q_dict))
            print("Not zeros in Q: " + str(not_zeros_in_q_dict))

        return q_dic


snake = game.Snake(width=100,height=100)
#snake = game.Snake()

#p = PLE(snake, fps=30, display_screen=False)
p = PLE(snake, fps=10, display_screen=True)

myAgent = MyAgent(p.getActionSet())


# try to get the memory info of the current process
#process = psutil.Process(os.getpid())
#print("PROZESS MEMORY INFO: ", process.memory_info().rss)

# save the q-dic in a file
# afile = open(os.getcwd() + '\q_dic.pkl', 'wb')
# pic.dump(my_q_dic, afile)
# afile.close()
#
# random_scores = myAgent.test_random(p, snake)
# learned_scores = myAgent.test(my_q_dic, p, snake)
#
# plt.ylabel("Number of games")
# plt.xlabel("Number of apples")
# plt.hist(random_scores, color='red')
# plt.hist(learned_scores, color='green')
# plt.show()

gamma = 0.5
learner = 0.7
explored = 0.4

out_file = open(os.getcwd() + '/q_dic600000.pkl', 'wb')
infile = open(os.getcwd() + '/q_dic500000.pkl', 'rb')
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
