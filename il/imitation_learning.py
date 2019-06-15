#%%
#%reload_ext autoreload
#%autoreload 2
#%matplotlib inline
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
from practical.raiRobot import RaiRobot
from practical.objectives import moveToPosition, gazeAt, scalarProductXZ, scalarProductZZ, distance
from practical.vision import findBallPosition, findBallInImage, getGraspPosition, maskDepth
from practical import utils
import libry as ry

#%%
gc.collect()
#%%
robot =  RaiRobot('', 'rai-robotModels/baxter/baxter_new.g')

#%%
names = robot.C.getJointNames()
#%%
right_joints = [name for name in names if name.startswith('right')]
#%%
len(right_joints)


#%%
robot.goHome()
#%%
def gatherDataSet(steps=10, pos = [0.7, 0, 1]):
    data = []
    for _ in range(steps):
        q_data = []
        robot.C.setJointState(robot.q_home)
        q_start = robot.C.getJointState(right_joints)
        q = robot.trackPath(pos, 'ball', 'baxterR', sendQ=True, joints=right_joints)
        q_dot_start=q[0] - q_start
        q_data.append(np.concatenate([q_start, q_dot_start]))
        for i in range(len(q)):
            if i < len(q)-1:
                q_dot = q[i + 1] - q[i]
                q_data.append(np.concatenate([q[i], q_dot]))
            else:
                q_data.append(np.concatenate([q[i], np.zeros(q[i].shape)]))
        data.append(q_data)
    return data


#%%
robot.closeBaxterR()
#%%
data = gatherDataSet()
#%%
data[0]

#%%
data = list(map(np.array, data))

#%%
data[0].shape
#%%
import pickle
#%%
with open('data/baxter.pkl' , 'wb') as f:
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
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
data = cleanFromGripperShit(data)
#%%
import pbdlib as pbd
#%%
model = pbd.HMM(nb_states=4, nb_dim=30)
#%%
model.init_hmm_kbins(data)
#%%
model.em(data, reg=1e-8)
#%%
robot.goHome(hard=True)

for i in range(100):
    time.sleep(1)
    q = robot.C.getJointState()

    q = np.delete(q, [15, 16])
    q_dot = model.predict(q, i)
    print(q_dot)
    q_new = q + q_dot
    q_new = np.concatenate([q_new, [0,0]])
    robot.move(q_new)

#%%
model.predict(data[0][0][:15], 0)







#%%
