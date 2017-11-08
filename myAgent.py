class MyAgent(object):
    def __init__(self,actionset):
        self.actionset = actionset
    def pickAction(self, reward, obs):
        #return self.actionset[0]
        return self.actionset[self.qLearningAction()]
        #return self.actionset[num.random.randint(0,4)]
    def qLearningAction(self):
        return 0