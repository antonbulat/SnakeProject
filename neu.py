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
        return self.actionset[num.randint(0, 5)]

    # Q-Leaning Algorithm
    # returns a q-dictionary
    def train(self, p):
        p.init()
        nb_frames = 1000
        # because of the number of states we work with a dictionary with (state,action) as key
        # so we always have to check if the key is already in the dictionary
        # - if not we have to initialize the key with 0
        q_dic = {}
        gamma = 0.75
        current_state = my_hash(p.getScreenRGB().tolist())  # make the screen hashable
        for f in range(nb_frames):
            if p.game_over():
                p.reset_game()
            # //////////////////////////////////////////////////////////////////////////////////////////
            else:
                action = self.pickRandomAction()  # select a random action --- good solution?
                reward = p.act(action)  # get reward
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
                # TODO das hier konvergiert noch nicht sondern wird einfach höher gesetzt...nicht so sinnvoll...
                q_dic[(current_state, action)] += (reward + gamma * q_dic[(next_state, max_action)])  # Bellman equation
                print("state, action, maxq",current_state,action,reward + gamma * q_dic[(next_state, max_action)])
                current_state = next_state
        return q_dic


snake = game.Snake(width=150, height=150, init_length=3)

p = PLE(snake, fps=30, display_screen=True)

myAgent = MyAgent(p.getActionSet())

start_time = time.time()
my_q_dic = myAgent.train(p)
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
afile = open(r'C:\Users\Julia\Documents\WS17-18\MLPraktikum\q_dic.pkl', 'wb')
pic.dump(my_q_dic, afile)
afile.close()
print("Q_DIC: ",my_q_dic)
