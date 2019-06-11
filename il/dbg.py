#%%
import gc
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
sys.path.append('.')
import time
import os
print(os.getcwd())
#%%
import pickle
#%%
with open('data/baxter.pkl' , 'rb') as f:
    data = pickle.load(f)
#%%

def cleanFromGripperShit(data):
    data_out = []
    index = [15, 16, 32, 33]
    for i, traj in enumerate(data):
        traj_out = []
        for j, q in enumerate(traj):
           q_done = np.delete(q, index)
           traj_out.append(q_done)
        data_out.append(np.array(traj_out))
    return data_out
#data = cleanFromGripperShit(data)
#%%fasdf
import pbdlib as pbd
#%%
model = pbd.HMM(nb_states=4, nb_dim=14)
#%%
model.init_hmm_kbins(data)
#%%
model.em(data, obs_fixed=True, left_to_right=False)
#%%
np.set_printoptions(precision=4)
for i in range(100):
    msg = model.predict(data[1][i][0:7], i)
    print("pr: ", msg)
    print("gt: ", data[1][i][7:])
    print('\n')

#%%

#%%
from practical.raiRobot import RaiRobot
from practical.objectives import moveToPosition, gazeAt, scalarProductXZ, scalarProductZZ, distance
from practical.vision import findBallPosition, findBallInImage, getGraspPosition, maskDepth
from practical import utils
from practical.dexnet import utils as dex_utils
import libry as ry

#%%
robot =  RaiRobot('', 'rai-robotModels/baxter/baxter_new.g')

#%%
names = robot.C.getJointNames()
#%%
right_joints = [name for name in names if name.startswith('right')]
#%%

#%%
np.set_printoptions(precision=4)
#%%
for i in range(100):
    q = robot.C.getJointState(right_joints)
    q_dot = model.predict(q, i)
    q_new = q_dot * 0.01 + q
    robot.C.setJointState(q_new, right_joints)

#%%
ik = robot.C.komo_IK(False)

#%%
for i in range(1000):
    ik.clearObjectives()
    q = robot.C.getJointState(right_joints)
    q_dot = model.predict(q, i)
    q_dot_new = np.zeros((17,))
    q_dot_new[[1, 3, 5, 7, 9, 11, 13]] = q_dot
    ik.addObjective(feature=ry.FS.qItself, type=ry.OT.eq, order=1, target=q_dot_new)
    ik.optimize(True)
    frames = ik.getConfiguration(0)
    robot.C.setFrameState(frames)

#%%
