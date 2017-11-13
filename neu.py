import ple.games as game
import numpy.random as num
import pickle as pic
import os
import psutil
import sys
import time
import itertools

from ple import PLE

# naive hash function for the screen
def my_hash(list3d):
    print("3DLIST: ",list3d)
    list2d = list(itertools.chain(*list3d))
    list1d = list(itertools.chain(*list2d))
    return hash(frozenset(list1d)) #  TODO Achtung das funktioniert nicht! Problem: durch frozenset wird reihnenfolge der elemente eliminiert -> keine funktionierende hashfunktion


class MyAgent(object):
    def __init__(self, actionset):
        self.actionset = actionset

    def pickRandomAction(self):
        #actions -> "up": K_w, "left": K_a, "right": K_d, "down": K_s, "none": -
        return self.actionset[num.randint(0, 4)]

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

    def pickBestActionRndomized(self, state, dict):
        actionsValues = dict[state]
        maxQ = max(actionsValues)
        count = actionsValues.count(maxQ)
        if count > 1:
            best = [i for i in range(len(actionsValues)) if actionsValues[i] == maxQ]
            i = num.random.rand(best)
            #random.choice(best)
        else:
            i = actionsValues.index(maxQ)

        return dict[state].__getitem__(i)

    def foodIsNearer(self, state1, state2):
        if (abs(state1["snake_head_x"]-state1["food_x"])+ abs(state1["snake_head_y"]-state1["food_y"])
            >abs(state2["snake_head_x"]-state2["food_x"])+ abs(state2["snake_head_y"]-state2["food_y"])):
            return True
        else:
            return False

    def getSmartReward(self, p, snake, current_state, next_state):
        if p.game_over:
            return snake.rewards["loss"]
        elif self.foodIsNearer(current_state, next_state):
            return snake.rewards["positive"]
        elif (self.foodIsNearer(current_state, next_state)):
            return snake.rewards["negative"]
        else:
            return snake.rewards["win"]

    # Q-Leaning Algorithm
    # returns a q-dictionary
    def train(self, p, snake):
        p.init()
        nb_frames = 100
        # because of the number of states we work with a dictionary with (state) as key
        # so we always have to check if the key is already in the dictionary
        # if not we have to initialize the key with Zeros

        q_dic = {snake.getGameState().values():[0, 0, 0, 0, 0]}
        gamma = 0.75
        current_state = snake.getGameState().values() #my_hash(p.getScreenRGB().tolist())  # make the screen hashable
        smartReward = snake.rewards["tick"]
        for f in range(nb_frames):
            if p.game_over():
                p.reset_game()
                # Bellman equation
                #q_dic[current_state] = reward + gamma * q_dic[next_state].__getitem__(self.pickBestAction(next_state, q_dic))
                q_dic[current_state] = smartReward + gamma * q_dic[next_state].__getitem__(self.pickBestAction(next_state, q_dic))
                smartReward = snake.rewards["tick"]
            # //////////////////////////////////////////////////////////////////////////////////////////
            else:
                if (num.randint(1, 10) % 10 == 0): # a little bit random inside would be better
                    action = self.pickRandomAction() # select a random action
                else:
                    action = self.pickBestActionRndomized(current_state, q_dic)  # select random max action value
                reward = p.act(action)  # get reward
                next_state = snake.getGameState().values()  # get next state
                if (next_state) not in q_dic:  # if next state is not in the dictionary
                    q_dic[(next_state)] = [0, 0, 0, 0, 0]  # add initial value
                smartReward = self.getSmartReward(p, snake, current_state, next_state)  # get smart reward (Manhatten)
                #  Bellman equation
                #q_dic[current_state] = (reward + gamma * q_dic[next_state].__getitem__(self.pickBestAction(next_state, q_dic)))
                q_dic[current_state] = (smartReward + gamma * q_dic[next_state].__getitem__(self.pickBestAction(next_state, q_dic)))
                print("state, action, maxq",current_state,action,reward + gamma * q_dic[next_state])
                current_state = next_state
        return q_dic


snake = game.Snake(width=250, height=250, init_length=3)

p = PLE(snake, fps=30, display_screen=True)

myAgent = MyAgent(p.getActionSet())

start_time = time.time()
my_q_dic = myAgent.train(p, snake)
end_time = time.time()

print("TIME OF Q_LEARNING", end_time - start_time)

# try to get the memory info of the current process
process = psutil.Process(os.getpid())
print("PROZESS MEMORY INFO: ", process.memory_info().rss)

# try to calculate the size of the q-dictionary... this does not work :(
#size = sys.getsizeof(my_q_dic)
#size += sum(map(sys.getsizeof, my_q_dic.itervalues())) + sum(map(sys.getsizeof, my_q_dic.iterkeys()))
#print("SIZE OF DICTIONARY: ", size)

# save the q-dic in a file
afile = open(os.getcwd() + '\q_dic.pkl', 'wb')
print(os.getcwd() + '\q_dic.pkl')
pic.dump(my_q_dic, afile, pic.HIGHEST_PROTOCOL)
afile.close()
print("Q_DIC: ",my_q_dic)
