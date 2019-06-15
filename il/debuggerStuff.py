import hmms
import pbdlib as pbd
import pickle
import numpy as np
import os
from collections import defaultdict
import sys
# add the folder where libry.so is located to the path. Otherwise the import will crash.
sys.path.append('../robotics-course/ry/')
import libry as ry
import time
os.getcwd()

# load names of joints to map from ros 2 rai joints. Only relevant joints (of right arm) are loaded
with open('bagfiles/names.pkl', 'rb') as f:
    names = pickle.load(f)

# load demonstration dataset type: List of np.array() with shape=(np_steps_in_trajectory, 2*dim(q))
with open('bagfiles/dataset.pkl', 'rb') as f:
    demos = pickle.load(f)

# initialize HMM model and train using expectation-maximization (already implemented in https://gitlab.idiap.ch/rli/pbdlib-python)
model = pbd.HMM(4, 14)
model.init_hmm_kbins(demos)
model.em(demos, left_to_right=True)


# clear views, config and operate by setting shared pointers to 0. Otherwise the notebook has to be restarted,
# which is pretty annoying.
C = 0
v = 0
B = 0
    
# initialize config
C = ry.Config()
v = C.view()
C.clear()
C.addFile('../robotics-course/rai-robotModels/baxter/baxter_new.g')

# add simulation. Note: if the string argument is not an empty string, a ROS node is started
# and the joint state topics of the real baxter are subscribed. This won't work if you can't connect to Baxter.
# In order to connect to Baxter, uncomment the next 2 lines and set the correct IP address:
#os.environ["ROS_MASTER_URI"] = "http://thecount.local:11311/"
#os.environ["ROS_IP"] = "<your-ip-address>"

B = C.operate('')
B.sync(C)
C.makeObjectsConvex()
ik = C.komo_IK(False)
ik.setConfigurations(C)
time_per_phase=10
steps_per_phase=300
path_planner = C.komo_path(1, steps_per_phase, time_per_phase, False)
path_planner.setConfigurations(C)
q_home = C.getJointState()
right_joints = [name for name in C.getJointNames() if name.startswith('right')]

def get_joint_mapping():
    mapping = defaultdict(dict)
    print("RAI")
    for i, name in enumerate(C.getJointNames()):
        if name.startswith('right'):
            print("{} : {}".format(i, name))
            mapping[name]['rai'] = i
    print("ROS")
    i = 0
    for _, name in enumerate(names):
        if name.startswith('right'):
            print("{} : {}".format(i, name))
            mapping[name]['ros']=i
            i += 1
    return mapping

mapping = get_joint_mapping()


# directly setting joint velocities is not standard with the rai framework; playing around with the komo optimizer
def model_step_using_opt(t, kappa_p=0.3, kappa_v=0.1, model=model):
    B.sync(C)
    ik_path = C.komo_path(1, 2, 1, False)
    ik_path.setConfigurations(C)
    q_rai, q_dot = C.getJointState_qdot()
    q = np.zeros((7, ))
    for name in right_joints:
        q[mapping[name]['ros']] = q_rai[mapping[name]['rai']]
    q_dot = model.predict_qdot(q, t)
    q_new = model.predict_q(q_dot, q, t)
    q_dot_rai = np.zeros((17,))
    q_ddot_rai = np.zeros((17,))
    q_ddot = 0.3 * (q - q_new)
    for name in right_joints:
        q_rai[mapping[name]['rai']] = q_new[mapping[name]['ros']]
        q_dot_rai[mapping[name]['rai']] = q_dot[mapping[name]['ros']]
        q_ddot_rai[mapping[name]['rai']] = q_ddot[mapping[name]['ros']]
    ik_path.clearObjectives()
    ik_path.addObjective(feature=ry.FS.qItself, order=0, target=q_rai, type=ry.OT.eq, scale=[1])
    ik_path.addObjective(feature=ry.FS.qItself, order=1, target=q_dot_rai, type=ry.OT.eq, scale=[.5])
    ik_path.addObjective(feature=ry.FS.qItself, order=2, target=q_ddot_rai, type=ry.OT.eq, scale=[.5])
    ik_path.optimize(False)
    C.setFrameState(ik_path.getConfiguration(1))
    q_rai = C.getJointState()
    B.move([q_rai], [1], True)
    B.wait()
    return np.concatenate([q_new, q_dot])

def plan_trajectory(model, kappa_p=0.3, steps=20, timePerPhase=1):
    B.sync(C)
    ik_path = C.komo_path(1, 2, timePerPhase, False)
    ik_path.setConfigurations(C)
    path = []
    times = []
    trajectory = []
    currentTime = 0
    for t in range(steps):
        q_rai, q_dot = C.getJointState_qdot()
        q = np.zeros((7, ))
        for name in right_joints:
            q[mapping[name]['ros']] = q_rai[mapping[name]['rai']]
        q_dot = model.predict_qdot(q, t)
        q_new = model.predict_q(q_dot, q, t)
        q_dot_rai = np.zeros((17,))
        q_ddot_rai = np.zeros((17,))
        q_ddot = 0.3 * (q - q_new)
        for name in right_joints:
            q_rai[mapping[name]['rai']] = q_new[mapping[name]['ros']]
            q_dot_rai[mapping[name]['rai']] = q_dot[mapping[name]['ros']]
            q_ddot_rai[mapping[name]['rai']] = q_ddot[mapping[name]['ros']]
        ik_path.clearObjectives()
        ik_path.addObjective(feature=ry.FS.qItself, order=0, target=q_rai, type=ry.OT.eq, scale=[1])
        ik_path.addObjective(feature=ry.FS.qItself, order=1, target=q_dot_rai, type=ry.OT.eq, scale=[.5])
        ik_path.addObjective(feature=ry.FS.qItself, order=2, target=q_ddot_rai, type=ry.OT.eq, scale=[.5])
        ik_path.optimize(False)
        C.setFrameState(ik_path.getConfiguration(1))
        q_rai = C.getJointState()
        path.append(q_rai)
        currentTime += timePerPhase
        times.append(currentTime)
        trajectory.append(np.concatenate([q, q_dot]))
    return (path, currentTime), trajectory




# get velocity and new joint configuration from HMM-model, directly set joint configurations
def model_step(t, model=model):
    B.sync(C)
    q_rai = C.getJointState()
    q = np.zeros((7, ))
    for name in right_joints:
        q[mapping[name]['ros']] = q_rai[mapping[name]['rai']]
    q_dot = model.predict_qdot(q, t)
    q_new = model.predict_q(q_dot, q, t)
    for name in right_joints:
        q_rai[mapping[name]['rai']] = q_new[mapping[name]['ros']]
    B.move([q_rai], [1], True)
    return np.concatenate([q_new, q_dot])

def run_model(n_steps, ik=False, model=model):
    B.moveHard(q_home)
    C.setJointState(q_home)
    trajectory = []
    for i in range(n_steps):
        time.sleep(0.1)
        if ik:
            trajectory += [model_step_using_opt(i)]
        else:
            trajectory += [model_step(i)]
    return trajectory

def run_demo(demo):
    q_rai = np.zeros((17,))
    for i, qqdot in enumerate(demo):
        for name in right_joints:
            q_rai[mapping[name]['rai']] = qqdot[mapping[name]['ros']]
        time.sleep(1)
        B.move([q_rai], [0.1], True)

trajectory = run_model(20, True)

def plan_demo(pos1, pos2, pos3, start=q_home):
    # first generate target frames
    if "ball1" not in C.getFrameNames():
        C.addObject(name="ball1", shape=ry.ST.sphere, size=[0.01], pos=pos1, color=[0, 0, 1], parent="base_footprint")
        C.addObject(name="ball2", shape=ry.ST.sphere, size=[0.01], pos=pos2, color=[0, 1, 0], parent="base_footprint")
        C.addObject(name="ball3", shape=ry.ST.sphere, size=[0.01], pos=pos3, color=[1, 0, 0], parent="base_footprint")
    C.frame("ball1").setPosition(pos1)
    C.frame("ball2").setPosition(pos2)
    C.frame("ball3").setPosition(pos3)
    path_planner = C.komo_path(1, 30, time_per_phase, False)
    # generate some kind of random start position
    noise = np.random.normal(0, 0.1, start.shape[0]-2)
    # don't change gripper values
    start[0:-2] += noise
    B.moveHard(start)
    B.sync(C)
    path_planner.setConfigurations(C)
    path_planner.clearObjectives()
    path_planner.addObjective(type=ry.OT.eq, feature=ry.FS.positionDiff, frames=["baxterR", "ball1"], time=[0.3])
    path_planner.addObjective(type=ry.OT.eq, feature=ry.FS.scalarProductYZ, frames=["baxterR", "ball1"], target=[0], time=[0.3])
    path_planner.addObjective(type=ry.OT.eq, feature=ry.FS.positionDiff, frames=["baxterR", "ball2"], time=[0.6])
    path_planner.addObjective(type=ry.OT.eq, feature=ry.FS.scalarProductYZ, frames=["baxterR", "ball2"], target=[0], time=[0.6])
    path_planner.addObjective(type=ry.OT.eq, feature=ry.FS.positionDiff, frames=["baxterR", "ball3"], time=[1.])
    path_planner.addObjective(type=ry.OT.eq, feature=ry.FS.scalarProductYZ, frames=["baxterR", "ball3"], target=[0], time=[1])
    path_planner.addObjective(type=ry.OT.sos, feature=ry.FS.qItself, time=[0, 1], target=start)
    path_planner.addObjective(type=ry.OT.sos, feature=ry.FS.qItself, time=[0,  1.], order=1)
    path_planner.optimize(False)
    t = path_planner.getT()
    demo = []
    q_old = np.zeros((7,))
    for name in right_joints:
        q_old[mapping[name]['ros']] = start[mapping[name]['rai']]
    q_ros = np.zeros((7,))
    path = []
    for i in range(t):
        C.setFrameState(path_planner.getConfiguration(i))
        q_new = C.getJointState()
        path += [q_new]
        for name in right_joints:
            q_ros[mapping[name]['ros']] = q_new[mapping[name]['rai']]
        q_dot = q_ros - q_old
        demo.append(np.concatenate([q_old, q_dot]))
        q_old = q_ros
    demo.append(np.concatenate([q_ros, np.zeros((7,))]))
    B.move(path, [time_per_phase/steps_per_phase * i for i in range(t)], False)
    return np.array(demo)


for i in range(3):
    demo = plan_demo([1.0, 0.3, 1], [0.0, 0.6, 0.5], [0.5, 0.4, 1.3])
    demos.append(demo)



model2 = pbd.HMM(7, 14)
model2.init_hmm_kbins(demos)
model2.em(demos, left_to_right=True)

trajectory = run_model(10, True, model=model2)
print("yoiu")