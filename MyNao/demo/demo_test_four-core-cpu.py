from speed_control import Speed_Controller, Speed_Controller_and_Motion_Actuator
from motion_select import Motion_Selector
from online_beat_extract import beat_extractor
from multiprocessing import Process, Queue
from angel_transmit import joint_actuate
from manage_joints import get_first_handles
import time
import numpy as np
import json
import os
import sim
import sys
"""
def beat_extractor(queue_beat):
    #print('Process to extract beats: %s' % os.getpid())
    time_init = time.time()
    while(True):
        time.sleep(0.5)
        queue_beat.put(time.time()-time_init)
        #print('beat_time!')

"""



def motion_selector(queue_motion_from_selector):
    #print('Process to select motion: %s' % os.getpid())
    ms = Motion_Selector('MyNao\\motion_base\\motion_base.json')
    ms.transmit_motion_info(queue_motion_from_selector)
    while(True):
        if queue_motion_from_selector.empty():
            ms.update_motion()
            ms.transmit_motion_info(queue_motion_from_selector)

def motion_actuator(queue_beat, queue_motion_from_selector):
    #print('Process to actuate motion: %s' % os.getpid())
    ip = '127.0.0.1'
    port = 19997
    sim.simxFinish(-1)  # just in case, close all opened connections
    clientID = sim.simxStart(ip, port, True, True, -5000, 5)
    # Connect to V-REP
    if clientID == -1:
        import sys
        sys.exit('\nV-REP remote API server connection failed (' + ip + ':' +
                    str(port) + '). Is V-REP running?')
    print('Connected to Remote API Server')  # show in the terminal
    Body = {}
    get_first_handles(clientID,Body)    #get first handles of Nao in the virtual environment
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

if __name__ == '__main__':
    #print('father:',  os.getpid())
    queue_beat = Queue(maxsize=1)
    queue_motion_from_selector = Queue(maxsize=1)

    process_beat_extract = Process(target=beat_extractor, args=(queue_beat,))
    process_motion_select = Process(target=motion_selector, args=(queue_motion_from_selector,))
    process_motion_actuate = Process(target=motion_actuator, args=(queue_beat, queue_motion_from_selector))
    
    process_beat_extract.start()
    process_motion_select.start()
    process_motion_actuate.start()

    process_beat_extract.join()
    process_motion_select.join()
    process_motion_actuate.join()

                
 