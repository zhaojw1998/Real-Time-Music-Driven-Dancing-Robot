#from speed_control import Speed_Controller, Speed_Controller_and_Motion_Actuator
#from motion_select import Motion_Selector
#from online_beat_extract import beat_extractor
import sys
sys.path.append('C:\\Users\\lenovo\\Desktop\\AI-Project-Portfolio\\Amendments\\scipts for graph construction\\simulation and check sequence validity')

from multiprocessing import Process, Queue
#from angel_transmit import init_joint
from angel_transmit import joint_actuate
from manage_joints import get_first_handles
import time
import numpy as np
import json
import os
import sim


if __name__ == '__main__':
    #print('Process to actuate motion: %s' % os.getpid())
    ip = '127.0.0.1'
    port = 19997
    sim.simxFinish(-1)  # just in case, close all opened connections
    clientID = sim.simxStart(ip, port, True, True, -5000, 5)
    # Connect to V-REP
    if clientID == -1:
        sys.exit('\nV-REP remote API server connection failed (' + ip + ':' +
                    str(port) + '). Is V-REP running?')
    print('Connected to Remote API Server')  # show in the terminal

    sim.simxStartSimulation(clientID, sim.simx_opmode_oneshot)

    Body = {}
    get_first_handles(clientID,Body)    #get first handles of Nao in the virtual environment

    with open('C:\\Users\\lenovo\\Desktop\\AI-Project-Portfolio\\primitive_transfer_graph.json', 'r') as f:
        primitive_transfer_graph = json.load(f)

    cluster_id = 1
    time1 = time.time()
    while(True):
        primitive_id = primitive_transfer_graph[str(cluster_id)]['include'][0]
        print(primitive_id)
        
        primitive = np.load('C:\\Users\\lenovo\\Desktop\\AI-Project-Portfolio\\danceprimitives\\'+primitive_id+'\\dance_motion_'+str(int(primitive_id))+'.npy')
        primitive = primitive.reshape(-1, 17, 3)

        motion = {}
        for i in range(primitive.shape[0]):
            motion[str(i)] = {}
            for j in range(primitive.shape[1]):
                motion[str(i)][str(j)] = [primitive[i][j][0], primitive[i][j][2], primitive[i][j][1]]
        
        for frame in range(primitive.shape[0]):
            joint_actuate(clientID, Body, motion, frame)
            returnCode, position=sim.simxGetObjectPosition(clientID, Body['HeadYaw'], -1, sim.simx_opmode_streaming)
            #print(position)
            time.sleep(0.1)

        rand = np.random.rand()
        margin = 0
        next_options = primitive_transfer_graph[str(cluster_id)]['lead_to']
        for key in next_options:
            margin += next_options[key]
            if rand < margin:
                cluster_id = int(key)
                break
        #if time.time()-time1 > 5:
        #    init_joint(clientID, Body, motion, frame)
        #    sim.simxStopSimulation(clientID, sim.simx_opmode_oneshot)
            #sim.simxFinish(clientID)
            

    sys.exit()
        
    





    motion_base_dir = 'MyNao\\motion_base\\motion_base.json'
    with open(motion_base_dir, 'r') as f:
        motion_base = json.load(f)
    sc = Speed_Controller_and_Motion_Actuator(alpha=0.9, motion_base=motion_base, kp=2e-4, ki=1e-6, kd=5e-4)  #2e-4, 1e-6, 1e-6
    #time_init = time.time()
    #time.sleep(5)
    while(True):
        if not queue_motion_from_selector.empty():
            current_motion_idx = queue_motion_from_selector.get()
            sc.set_current_motion(current_motion_idx)
            current_motion = motion_base[str(current_motion_idx)]['frame']
            for frame in range(len(current_motion)):
                time_1=time.time()
                joint_actuate(clientID, Body, current_motion, frame)
                time_transmit = time.time()-time_1
                sc.set_current_frame(frame)
                sc.control(queue_beat)
                time_sleep = sc.get_time_sleep()
                if time_sleep < 0.02:
                    time.sleep(0.033)
                else:
                    time.sleep(max(0.02, time_sleep - time_transmit))

 