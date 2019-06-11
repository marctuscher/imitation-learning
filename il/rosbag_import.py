#%%
%reload_ext autoreload
%autoreload 2
%matplotlib inline
#%%
import rosbag
import pathlib
import os
import numpy as np
#%%
p = pathlib.Path('/home/niklas/git/uni/imitation-learning/bagfiles')
os.chdir(p)
files = os.listdir(p)
#%%
files
#%%
bag = rosbag.Bag(files[0])
#%%
msg_iter = bag.read_messages()
#%%

demos = []
#%%
for fname in files:
    if fname == "dataset.pkl":
        continue
    bag = rosbag.Bag(fname)
    tmp = []
    for _, msg, _ in bag.read_messages():
        pos = msg.position[9:16]
        vel = msg.velocity[9:16]
        tmp.append(np.concatenate([np.array(pos), np.array(vel)]))
    demos.append(np.array(tmp))
#%%
len(demos[3])
#%%
import pickle
#%%
with open("/home/niklas/git/uni/imitation-learning/bagfiles/dataset.pkl", "wb") as f:
    pickle.dump(demos, f, pickle.HIGHEST_PROTOCOL)

#%%
import pbdlib as pbd

#%%
model = pbd.HMM(7, 14)

#%%
model.init_hmm_kbins(demos)

#%%
model.em(demos, obs_fixed=False, left_to_right=True)

#%%
np.set_printoptions(precision=6)

#%%
model.plot()
#%%
for i in range(600):
    msg = model.predict(demos[1][i][0:7], i)
    print("pr: ", msg)
    print("gt: ", demos[1][i][7:])
    print('\n')

#%%

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
from collections import defaultdict
#%%
_, msg, _ = next(msg_iter)
names = msg.name
mapp = defaultdict(dict)
#%%
print("RAI")
for i, name in enumerate(C.getJointNames()):
    if name.startswith('right'):
        print("{} : {}".format(i, name))
        mapp[name]['rai'] = i

#%%
print("ROS")
i = 0
for j, name in enumerate(names):
    if name.startswith('right'):
        print("{} : {}".format(i, name))
        mapp[name]['ros']=i
        i += 1
#%%
mapp
#%%
right_joints = [name for name in C.getJointNames() if name.startswith('right')]
#%%
q = C.getJointState()
q_new = np.zeros((7,))
#%%
for name in right_joints:
    q_new[mapp[name]['ros']] = q[mapp[name]['rai']]


#%%
len(q_new)


#%%
q_dot_new = model.predict(q_new, 0)


#%%
q_dot_rai = np.zeros((17,))
for name in right_joints:
    q_dot_rai[mapp[name]['rai']] = q_dot_new[mapp[name]['ros']]
#%%
ik = C.komo_IK(False)

#%%
ik.clearObjectives()
ik.addObjective(type=ry.OT.eq, order=1, feature=ry.FS.qItself, target=q_dot_rai)


#%%
ik.optimize(True)

#%%
frames = ik.getConfiguration(0)
C.setFrameState(frames)

#%%

#%%
C.setJointState(q)

#%%
