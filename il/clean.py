#%%
%reload_ext autoreload
%autoreload 2
%matplotlib inline
#%%
os.environ["ROS_MASTER_URI"] = "http://thecount.local:11311/"
os.environ["ROS_IP"] = "129.69.216.204"
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
os.chdir('../../robotics-course')
#%%
import ry.libry as ry

#%%
gc.collect()
#%%
C = ry.Config()
v = C.view()
C.clear()
C.addFile('rai-robotModels/baxter/baxter_new.g')
B = C.operate('marcsNode')
B.sync(C)
#%%
q_home = C.getJointState()
#%%
C.getJointNames()
#%%
B.sendToReal(True)
#%%
B.moveHard(q_home)
#%%
B.sendToReal(False)
#%%
ball = C.addObject(parent="base_footprint",name="ball", shape=ry.ST.sphere, size=[.01], pos=[0.2,0.5,1], color=[0.,0.,1.])
#%%
frames = C.getFrameState()
C.setFrameState(frames)
#%%
path = C.komo_path(1, 5, 10, False)
#%%
path.addObjective(type=ry.OT.eq, feature=ry.FS.scalarProductYZ, frames=['baxterR', 'ball'], target=[0], time=[])
path.addObjective(type=ry.OT.eq, feature=ry.FS.scalarProductZZ, frames=['baxterR', 'ball'], target=[1], time=[])
#path.addObjective(type=ry.OT.eq, feature=ry.FS.distance, frames=['baxterR', 'ball'], target=[-0.2], time=[0.8])
path.addObjective(type=ry.OT.eq, feature=ry.FS.positionDiff, frames=['baxterR', 'ball'], time=[1.0])
path.addObjective(type=ry.OT.sos, feature=ry.FS.qItself, frames=[], order=1, time=[1.0])
#%%
path.setConfigurations(C)
#%%
path.optimize(False)
#%%
frames = path.getConfiguration(3)
#%%
C.setFrameState(frames)
#%%
poses = []
t = path.getT()
for i in range(t):
    frames = path.getConfiguration(i)
    C.setFrameState(frames)
    poses += [C.getJointState()]
#%%
def gatherDataSet(steps=10, pos = [0.7, 0.5, 1]):
    data = []
    for _ in range(steps):
        q_data = []
        robot.goHome(hard=True, randomHome=True)
        q_start = robot.C.getJointState()
        q = robot.trackPath(pos, 'ball', 'baxterR', sendQ=True)
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
data = list(map(np.array, data))
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
model = pbd.HMM(nb_states=7, nb_dim=30)
gmr = pbd.GMR
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
