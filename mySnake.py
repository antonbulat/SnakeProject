import ple.games as game
import numpy as num
import myAgent as myA
from ple import PLE



snake=game.Snake(width=640,height=640, init_length=3)


p = PLE(snake,fps=30, display_screen=True, force_fps=False)
p.init()

myAgent = myAgent(p.getActionSet())

nb_frames = 1000
reward = 0.0

for f in range(nb_frames):
    if p.game_over(): #check if the game is over
        p.reset_game()

    obs = p.getScreenRGB()
    action = myAgent.pickAction(reward, obs)
    reward = p.act(action)