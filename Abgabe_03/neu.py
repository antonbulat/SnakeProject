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
        self.nb_frames = 1000

    def pickRandomAction(self):
        #actions -> "up": K_w, "left": K_a, "right": K_d, "down": K_s, "none": -
        return self.actionset[num.randint(0, 4)]

    #TODO: diese Funktion muss noch angepasst werden auf die Dictionary Struktur
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

    def test_random(self, p, snake):
        scores = []
        p.init()
        #nb_frames = 100
        for f in range(self.nb_frames):
            if p.game_over():
                scores.append(snake.getScore() + 5)  # save the scores
                p.reset_game()
            else:
                max_action = self.pickRandomAction()  # take a random action
                p.act(max_action)  # execute action
        return scores

    def test(self, q_dic, p, snake):
        scores = []
        p.init()
        #nb_frames = 100
        current_state = my_hash(p.getScreenRGB())  # make the screen hashable
        for f in range(self.nb_frames):
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

    def getSmartReward(self, isGameOver, snake, current_state, next_state):
        if isGameOver:
            return snake.rewards["loss"]
        elif self.foodIsNearer(current_state, next_state):
            return snake.rewards["positive"]
        elif (self.foodIsNearer(current_state, next_state)):
            return snake.rewards["negative"]
        else:
            return snake.rewards["win"]

    def foodIsNearer(self, state1, state2):
        if (abs(state1["snake_head_x"] - state1["food_x"]) + abs(state1["snake_head_y"] - state1["food_y"])
                > abs(state2["snake_head_x"] - state2["food_x"]) + abs(state2["snake_head_y"] - state2["food_y"])):
            return True
        else:
            return False

    # Q-Leaning Algorithm
    # returns a q-dictionary
    def train(self, p, snake):
        p.init()
        #nb_frames = 100
        # because of the number of states we work with a dictionary with (state,action) as key
        # so we always have to check if the key is already in the dictionary
        # - if not we have to initialize the key with 0
        q_dic = {}
        gamma = 0.75
        learn = 0.85  # controls how much difference between previous Q-value
        # and newly proposed Q-value is taken into account
        current_state = my_hash(p.getScreenRGB().tolist())  # make the screen hashable
        current_state_snake = snake.getGameState()
        action = self.pickRandomAction()  # select a random action
        for f in range(self.nb_frames):
            if p.game_over():
                p.reset_game()
            # //////////////////////////////////////////////////////////////////////////////////////////
            else:
                reward = p.act(action)  # get reward
                bool = p.game_over()
                next_state_snake = snake.getGameState()
                smartReward = self.getSmartReward(bool, snake, current_state_snake, next_state_snake)
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
                #q_dic[(current_state, action)] += learn * (reward + gamma * q_dic[(next_state, max_action)]- q_dic[(current_state, action)])
                #We are using SmartReward now. This means we use the information of the snake and food position.
                q_dic[(current_state, action)] += learn * (smartReward + gamma * q_dic[(next_state, max_action)]- q_dic[(current_state, action)])

                current_state = next_state
                current_state_snake = next_state_snake
                if num.randint(0, 1) == 0:
                    action = max_action  # to explore new better states with probability 0.5
                else:
                    action = self.pickRandomAction()
        return q_dic


snake = game.Snake()

p = PLE(snake, fps=30, display_screen=True)

myAgent = MyAgent(p.getActionSet())

start_time = time.time()
my_q_dic = myAgent.train(p, snake)
end_time = time.time()

print("TIME OF Q_LEARNING", end_time - start_time)

# try to get the memory info of the current process
process = psutil.Process(os.getpid())
print("PROZESS MEMORY INFO: ", process.memory_info().rss)

# save the q-dic in a file
afile = open(os.getcwd() + '\q_dic.pkl', 'wb')
pic.dump(my_q_dic, afile)
afile.close()

random_scores = myAgent.test_random(p, snake)
learned_scores = myAgent.test(my_q_dic, p, snake)

plt.ylabel("Number of games")
plt.xlabel("Number of apples")
plt.hist(random_scores, color='red')
plt.hist(learned_scores, color='green')
plt.show()
