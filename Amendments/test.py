#from speed_control import Speed_Controller, Speed_Controller_and_Motion_Actuator
#from motion_select import Motion_Selector
#from online_beat_extract import beat_extractor

from multiprocessing import Process, Queue
#from angel_transmit import joint_actuate
#from manage_joints import get_first_handles
import time
import numpy as np
import json
import os
import pickle
#import sim


if __name__ == '__main__':
    #print('Process to actuate motion: %s' % os.getpid())
    """
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
    """
    with open('C:\\Users\\lenovo\\Desktop\\AI-Project-Portfolio\\primitive_transfer_graph.json', 'r') as f:
        primitive_transfer_graph = json.load(f)

    cluster_id = 1
    time1 = time.time()
    record = []
    while(len(record)<348):
        primitive_id = primitive_transfer_graph[str(cluster_id)]['include'][0]
        record.append(int(primitive_id))
        print(primitive_id)
        #print(primitive_id)
        rand = np.random.rand()
        margin = 0
        next_options = primitive_transfer_graph[str(cluster_id)]['lead_to']
        for key in next_options:
            margin += next_options[key]
            if rand < margin:
                cluster_id = int(key)
                break
        #if time.time()-time1 > 5:
        #    sim.simxStopSimulation(clientID, sim.simx_opmode_oneshot)
            #sim.simxFinish(clientID)
    print(record)
    #with open('pro-record8.txt', 'wb') as f:
    #    pickle.dump(record, f, protocol=2)