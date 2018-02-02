import tensorflow as tf
from ple import PLE
import ple.games as game
import random as random
import collections as col
import numpy as num
import time
import cv2
import os
import math
import pygame as pg
import sys

# Tensorflow functions :
# Set random values from a truncated normal distribution
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

# Good practice to initialize a slightly positive bias to avoid "dead neurons"
def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)

# A unit employing the rectifier is also called a rectified linear unit (ReLU). f(x)=max(0,x)
# A smooth approximation to the rectifier is the analytic function f(x)=log(1+e^x) which is called the softplus function
def relu(input, w, st, b):
    return tf.nn.relu(tf.nn.conv2d(input, w, strides=st, padding="SAME") + b)

# Max pooling uses the maximum value from each of a cluster of neurons at the prior layer
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Multiplies matrix a by matrix b, producing a * b
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


    hidden_conv_l3_flat = tf.reshape(hidden_max_pooling_l3, [-1, 256])#todo 320 values?!----256

    final_hidden_activations = tf.nn.relu(matmul(hidden_conv_l3_flat, W_feed_forward1,b_feed_forward1))
    output_l = matmul(final_hidden_activations, W_feed_forward2, b_feed_forward2)
    return input_l, output_l


def init_phi_function(phi_t, xt):
    _, screen_resized_binary = cv2.threshold(xt, 26, 255, cv2.THRESH_BINARY)
    state = num.stack(tuple(screen_resized_binary for _ in range(4)), axis=2)
    return state
def phi_function(phi_t, xt):
    _, screen_resized_binary = cv2.threshold(xt, 26, 255, cv2.THRESH_BINARY)
    transform= num.reshape(screen_resized_binary,(80,80,1))
    result=num.append(phi_t[:, :, 1:], transform, axis=2)
    return result



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

    def test_random(self, p, snake, time_sec):
        start_time = time.time()
        scores = []
        p.init()
        while (time.time() - start_time < time_sec):
            if p.game_over():
                scores.append(snake.get_fixed_score() + 5)  # save the scores
                p.reset_game()
            else:
                max_action = self.pickRandomAction()  # take a random action
                p.act(max_action)  # execute action
        return scores

    def test(self, p, snake, time_sec, session, input_l, output_l):

        start_time = time.time()
        scores = []
        p.init()
        #todo first 4 actions are random for initialisation! how to fix this?!
        xt = p.getScreenGrayscale()

        phi_t = None
        # execute the first step for initialisation of phi t
        phi_t = init_phi_function(phi_t, xt)
        action_index = random.randint(0, 4)
        p.act(self.actionset[action_index])
        xt = p.getScreenGrayscale()

        # execute the next 4 steps for initialisation of phi t and phi t+1
        for i in range(0, 4):
            if p.game_over():
                p.reset_game()  # do not save end images
                xt = p.getScreenGrayscale()
            phi_t = init_phi_function(phi_t, xt)
            action_index = random.randint(0, 4)
            p.act(self.actionset[action_index])
            xt = p.getScreenGrayscale()

        # so now both deques have length 4... we could start training
        while(time.time()-start_time<time_sec):
            if p.game_over():
                scores.append(snake.get_fixed_score() + 5)  # save the scores
                p.reset_game()
                xt = p.getScreenGrayscale()

            phi_t = phi_function(phi_t, xt)
            action_index=num.argmax(session.run(output_l, feed_dict={input_l:[phi_t]})[0])#get index of best action
            #print("Action index: ",action_index,session.run(output_l,feed_dict={input_l:[phi_t]}))
            p.act(self.actionset[action_index])  # execute action
            xt=p.getScreenGrayscale()

        return scores

    def getSmartReward(self, current_state_snake, next_state_snake, reward, maxPosReward, maxNegReward):
        if reward == -5:  # game over
            return 0
        elif reward == 1:  # Snake found apple
            return 1
        else:
            dis_to_apple=math.sqrt((next_state_snake["snake_head_x"] - next_state_snake["food_x"])*
                                   (next_state_snake["snake_head_x"] - next_state_snake["food_x"])+
                                   (next_state_snake["snake_head_y"] - next_state_snake["food_y"])*
                                   (next_state_snake["snake_head_y"] - next_state_snake["food_y"]))
            if self.foodIsNearer(current_state_snake, next_state_snake):  # snake goes to apple
                return maxPosReward*(1-dis_to_apple/114)
            return maxNegReward*(1-dis_to_apple/114)  # snake goes away from apple
    '''

    def getSmartReward(self, current_state_snake, next_state_snake, reward, maxPosReward, maxNegReward):
        if reward == -5:  # game over
            return 0
        elif reward == 1:  # Snake found apple
            return 1
        else:
            if self.foodIsNearer(current_state_snake, next_state_snake):  # snake goes to apple
                return maxPosReward * (1 - 1 / (abs(next_state_snake["snake_head_x"] - next_state_snake["food_x"]) + abs(next_state_snake["snake_head_y"] - next_state_snake["food_y"])))
            return maxNegReward * (1 - (width + hight))/ (abs(next_state_snake["snake_head_x"] - next_state_snake["food_x"]) + abs(next_state_snake["snake_head_y"] - next_state_snake["food_y"]))  # snake goes away from apple
'''
    def foodIsNearer(self, state1, state2):
        return (abs(state1["snake_head_x"] - state1["food_x"]) + abs(state1["snake_head_y"] - state1["food_y"])
                > abs(state2["snake_head_x"] - state2["food_x"]) + abs(state2["snake_head_y"] - state2["food_y"]))


    # deep Q-Leaning Algorithm
    # returns a trained neuronal network
    def train(self, p, snake, loadNetwork, saveNetwork, network_path, time_sec,constantRandomizedTime, randomizedFrames,
              gamma,explored,screen_size_x=80,screen_size_y=80,mini_batch_size=16,maxPosReward=0.5,maxNegReward=0.1):

        # split the time in three parts...
        time_random = 60 * 30                   # 0,5 h fast komplett Random
        time_rand_network = (time_sec // 6) * 4 # 2/3 der Zeit minus 1h expored Network
                                                # Rest der Zeit semi Network (Fast nur Netzwerk)
        start_time = time.time()

        # init game
        p.init()
        nb_actions = 5  # number of possible actions
        input_frames = 4  # number of frames to consider as input
        session = tf.Session()
        # Create neuronal network with 3 layers
        input_l, output_l = init_neural_network(screen_size_x, screen_size_y, nb_frames=input_frames,nb_actions= nb_actions)
        action = tf.placeholder("float", [None, nb_actions])
        target = tf.placeholder("float", [None])
        readout_action = tf.reduce_sum(tf.multiply(output_l, action), reduction_indices=1)
        cost = tf.reduce_mean(tf.square(target - readout_action))
        train_operation = tf.train.AdamOptimizer(1e-6).minimize(cost)
        session.run(tf.global_variables_initializer())

        #///////////////////////////////////////////////////////////
        #Restore checkpoints

        if loadNetwork:
            if not os.path.exists(network_path):
                os.mkdir(network_path)

            saver = tf.train.Saver()
            checkpoint = tf.train.get_checkpoint_state(network_path)

            if checkpoint and checkpoint.model_checkpoint_path:
                saver.restore(session, checkpoint.model_checkpoint_path)
                print("Loaded checkpoints %s" % checkpoint.model_checkpoint_path)#TODO change string

        #//////////////////////////////////////////////////////////////
        # init replay memory as set
        replay_memory = col.deque(maxlen=500000)
        current_state_snake = snake.getGameState()

        xt= p.getScreenGrayscale()
        phi_t=None
        # execute the first step for initialisation of phi t
        phi_t=init_phi_function(phi_t,xt)

        action_index = random.randint(0, 4)
        p.act(self.actionset[action_index])
        xt = p.getScreenGrayscale()

        phi_tp1 = phi_function(phi_t, xt)

        # so now both deques have length 4... we could start learning
        t=0
        goon = True
        progress = 0
        result = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0]
        learned_scores = []
        timer_tmp = 1
        learning_results = open('learning_results_small_new.txt', 'w+')
        tmp_scores = col.deque(maxlen=200)
        print("Training wurde zu ", progress, "% absolviert")
        while(time.time()-start_time<time_sec and goon):
            t+=1
            events = pg.event.get()
            for event in events:
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_SPACE:
                        goon = False
            if p.game_over():
                tmp_scores.append(snake.get_fixed_score() + 5)
                if time.time() - start_time > (time_sec - 60 * 60):  #
                    learned_scores.append(snake.get_fixed_score() + 5)  # save the scores
                p.reset_game()
                xt = p.getScreenGrayscale()
                current_state_snake = snake.getGameState()

            action_array = num.zeros([nb_actions])

            if(time.time()-start_time<time_random and not constantRandomizedTime ): #TODO: Muss evtl. komplett auf eine Konstante umgestellt werden.
                if (random.random() <= 0.1):
                    arr=session.run(output_l,feed_dict={input_l:[phi_t]})[0]
                    action_index = num.argmax(arr)
                else:
                    action_index = random.randint(0,4)
            elif(constantRandomizedTime and t<randomizedFrames):
                if (random.random() <= 0.1):
                    arr=session.run(output_l,feed_dict={input_l:[phi_t]})[0]
                    action_index = num.argmax(arr)
                else:
                    action_index = random.randint(0,4)
            elif(time.time()-start_time<time_rand_network):
                if (random.random() <= explored):
                    arr=session.run(output_l,feed_dict={input_l:[phi_t]})[0]
                    action_index = num.argmax(arr)
                else:
                    action_index = random.randint(0,4)
            else:
                if (random.random() <= 0.95):
                    arr=session.run(output_l,feed_dict={input_l:[phi_t]})[0]
                    action_index = num.argmax(arr)
                else:
                    action_index = random.randint(0,4)

            #Resultate abspeichern
            if time.time()-start_time > 60*4*timer_tmp:
                myres=""
                for i in range(len(tmp_scores)-1):
                    myres+=str(tmp_scores[i])+","
                myres+=str(tmp_scores[len(tmp_scores)-1])+";"+str(timer_tmp)+"\n"
                learning_results.write(myres)
                learning_results.flush()
                timer_tmp+=1

            action_array[action_index] = 1
            reward = p.act(self.actionset[action_index])
            next_state_snake= snake.getGameState()
            xt1 = p.getScreenGrayscale()

            phi_t=phi_function(phi_t,xt)
            phi_tp1=phi_function(phi_tp1,xt1)

            smart_reward= self.getSmartReward(current_state_snake, next_state_snake, reward, maxPosReward, maxNegReward)
            replay_memory.append((phi_t,action_array,smart_reward,phi_tp1))
            # sample random minibatch of transitions from replay memory
            # test diffrent configurations
            if len(replay_memory)>mini_batch_size:
                mini_batch = random.sample(list(replay_memory), mini_batch_size)

                # mini_batch variables:
                yj = [] # expected rewards
                prev_states = [d[0] for d in list(mini_batch)]
                actions = [d[1] for d in list(mini_batch)]
                curr_states = [d[3] for d in list(mini_batch)]
                #reward_a = output_l.eval(feed_dict={input_l: curr_states},session=session)
                reward_a = session.run(output_l,feed_dict={input_l: curr_states})

                for j in range(0,mini_batch_size):
                    sequence=mini_batch[j]
                    if sequence[2]==0:# if reward is 0 next state is terminal state
                        yj.append(sequence[2])
                    else:
                        yj.append(sequence[2]+gamma*num.max(reward_a[j]))
                # gradient descent step

                train_operation.run(feed_dict={input_l: prev_states, action: actions, target: yj},session=session)

                if saveNetwork:
                    # save checkpoints for later
                    if t % 5000 == 0:
                        saver.save(session, network_path + '/network', global_step=t)
                        print("Session saved at: ",time.time()-start_time)

            current_state_snake=next_state_snake
            xt=xt1
            if (int(round(100*(time.time()-start_time)/time_sec))%5==0):
                if (progress!=int(round(100*(time.time()-start_time)/time_sec))):
                    progress = int(round(100*(time.time()-start_time)/time_sec))
                    print("Training wurde zu ",progress,"% absolviert")

        for elem in learned_scores:
            result[int(elem)] += 1
        print("Result of training:")
        for i in range(0, len(result) - 1):
            print(result[i], end=";")

        print(result[len(result) - 1])
        print()
        learning_results.close()

        return session, input_l, output_l


def main():
    #######+++++++!!!!!!!Variablen und ihre Wirkung auf das Program!!!!!!!+++++++#######

    ###Spielumgebung###
    #Die Breite und die Höhe des Fensters sollen bei 80!!! bleiben. Falls jedoch ein anderer Wert genutzt wird,
    #dann muss das Netzwerk umstrukturiert werden
    width=80                #Die Breite des Fensters und somit das Spielfeld auf dem Snake gespielt wird
    height=80               #Die Höhe des Fensters und somit das Spielfeld auf dem Snake gespielt wird
    fastTesting=True        #Um bei den Tests das Fenster für menschliches Auge ausreichend langsam zu machen,
                            #muss der Wert auf False gesetzt werden
    displayScreen=True      #Um den Trainingprozess und das Testing effizienter zu machen (mehr Berechnungen
                            #in gleicher Zeit), muss der Wert auf False gesetzt werden, was jedoch die Beobachtung
                            #des Prozesses während der Ausführung nicht mehr ermöglicht
    #Des Weiteren können Eigenschaften in der Klasse Snake angepasst werden. Zu diesen gehören unter anderem Größe und
    #Geschwindigkeit der Schlange, Größe und Positionirungseinschränkung des Apfels, die Farben der Objekte und des
    #Hintergrunds, und vieles mehr. Aktuell (16.01.18) wird die Klasse Schlange lockal mit unterschiedlichen Versionen
    #genutzt. Um zu vergleichbaren Ergebnissen zu kommen wird sich auf folgende Werte in der Snake Klasse geeinigt:
    #self.speed=0.45; player_width=0.05; food_width=0.09; Faktor für den Abstand vom Apfel zum Spielfeldrand = 2;
    #player_color=(100,255,100); food_color=(255,100,100); BG_COLOR=(25, 25, 25). Sondersituation Stand 2.2.18 Schlage
    #wächst nicht!

    ###Training###
    wantDebug=False             #Für das anzeigen von Details der einzelnen Variablen wärend der Berechnungen,
                                #muss dieser Wert auf True gesetzt werden
    loadNetwork=False           #Bei True wird das Netzwerk aus dem unten angegebenen Pfad geladen
    saveNetwork=False           #Bei True wird das Netzwerk in dem unten angegebenen Pfad gespeichert
    networkPath="network_"      #Pfad, in dem bzw. von dem das Netzwerk gespeichert bzw. geladen wird
    trainingTime=60*60*10       #Gesamtzeit die verwendet wird, um das Neuronale Netzwerk zu trainieren
                                #(nur Zufall / Zufall und Netzwerk / nur Netzwerk)
    testingTime=60*5            #Zeit, um das antrainierte Neuronale Netzwerk zu testen (Default = 5 min)
    constantRandomTime=False    #Zu Beginn des Trainings wird die Schlange zufällig gesteuert. Wird der Wert hier auf
                                #True gesetzt, so ist die Anzahl der zufälligen Aktionen zu Beginn fest, sonst beträgt
                                #diese ein Drittel der Trainingszeit
    randomizedFrames=10000      #Anzahl der zu Beginn ausgeführten Zufallsaktionen, wenn dieser Wert konstant sein soll
    gamma=0.5                   #Lernrate, die das erlernen neuer Erkenntnisse beeinflusst
    explored=0.75               #Zufallsrate, die die Wahl einer Zufälligen Aktion der Schlange beeinflusst
    miniBatchSize=200           #Anzahl der Zustände, die aus der Vergangenheit betrachtet werden
    maxPosReward=0.8            #Die maximale Belohnung, wenn die Schlange sich zum Apfel bewegt
    maxNegReward=0.1            #Die maximale Belohnung, wenn die Schlange sich weg vom Apfel bewegt




    snake = game.Snake(width=width, height=height)# DO NOT CHANGE SIZE!... network fails if you do...Should be 80x80!

    p = PLE(snake, fps=30, force_fps=fastTesting, display_screen=displayScreen) #Für Aufnahmen force_fps = False setzen

    myAgent = MyAgent(p.getActionSet(), wantDebug)


    #input = sys.argv[1]
    #gamma = float(input[0]) / 10
    #learner = float(input[1]) / 10
    #explored = float(input[1]) / 10

    startTrainingTime = time.time()
    session, input_l, output_l = myAgent.train(p,snake,loadNetwork,saveNetwork,networkPath,trainingTime,
                                               constantRandomTime, randomizedFrames,
                                               gamma,explored,screen_size_x=80,screen_size_y=80,
                                               mini_batch_size=miniBatchSize,maxPosReward=maxPosReward,
                                               maxNegReward=maxNegReward)
    endTrainingTime = time.time()
    print("TIME OF Q_LEARNING TRAINING: ", endTrainingTime - startTrainingTime)
    print("testing...")
    learned_scores = myAgent.test( p, snake, time_sec=testingTime, session=session, input_l=input_l, output_l=output_l)
    #learned_scores=myAgent.test_random(p,snake,time_sec=60*5)

    result = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    for elem in learned_scores:
        result[int(elem)] += 1
    for i in range(0,len(result)-1):
        print(result[i],end=";")

    print(result[len(result)-1])
    print("training time:",int(endTrainingTime-startTrainingTime),"; testing time:",testingTime,
          "; learning rate:",gamma,"; exploration rate:",explored,"; batch size:",miniBatchSize,
          "; positiv reward:",maxPosReward,"; negativ reward:",maxNegReward,";")


if __name__ == '__main__': main()
