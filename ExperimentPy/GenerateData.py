import numpy as np
from gym import utils
from ICCA_mujoco_env.env.hand import HandEnv
import sys
import gym
import torch
import pandas as pd


def Main(n=1000,testTrials=1,trainTestSplit=0.7,step=30):

    for j in range(testTrials):

        env=gym.make("Hand")
        final = np.array([])
        initial = np.array([])
        for _ in range(int(n*trainTestSplit)):
            env.reset()
            for i in range(step):
                if i ==1:
                    initial = np.append(initial,obs)
                obs,_,_,_,_= env.step(np.array([-1.5,-0.5,-3,-1.8,-0.6,-3,-1.8,-0.6,-3,-1.8,-0.6,-3,-1.8,-0.6,-0.5,-0.5,0,0.5,1,-0.25,0,0,0,0]))
            final = np.append(final,obs)

        final = np.reshape(final,(int(n*trainTestSplit),24))
        initial = np.reshape(initial,(int(n*trainTestSplit),24))

        dataPast = pd.DataFrame(initial)
        dataFuture = pd.DataFrame(final)

        dataPast.to_csv("ICCA.jl/data/pastTrainData"+str(j+1)+".csv",index=False)
        dataFuture.to_csv("ICCA.jl/data/futureTrainData"+str(j+1)+".csv",index=False)

        env=gym.make("Hand")
        final = np.array([])
        initial = np.array([])
        for _ in range(int(n*(1-trainTestSplit))):
            env.reset()
            for i in range(step):
                if i ==1:
                    initial = np.append(initial,obs)
                obs,_,_,_,_= env.step(np.array([-1.5,-0.5,-3,-1.8,-0.6,-3,-1.8,-0.6,-3,-1.8,-0.6,-3,-1.8,-0.6,-0.5,-0.5,0,0.5,1,-0.25,0,0,0,0]))
            final = np.append(final,obs)

        final = np.reshape(final,(int(n*(1-trainTestSplit)),24))
        initial = np.reshape(initial,(int(n*(1-trainTestSplit)),24))

        dataPast = pd.DataFrame(initial)
        dataFuture = pd.DataFrame(final)

        dataPast.to_csv("ICCA.jl/data/pastTestData"+str(j+1)+".csv",index=False)
        dataFuture.to_csv("ICCA.jl/data/futureTestData"+str(j+1)+".csv",index=False)
    return



if __name__ == "__main__":
    arguments = {'n':'1000' ,'testTrials':'3','trainTestSplit':'0.7','step':'30'}
    if len(sys.argv) <= 5:
        for i in sys.argv:
            split = i.split('=')
            arguments[split[0]] = split[1]
    Main(int(arguments['n']),int(arguments['testTrials']),int(arguments['trainTestSplit']),int(arguments['step']))