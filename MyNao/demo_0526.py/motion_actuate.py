from angel_transmit import Joint_Actuator
from manage_joints import get_first_handles
import time
import numpy as np
import json
import os
import sim
import sys

class Motion_Actuator(object):
    def __init__(self):
        
        ip = '127.0.0.1'
        port = 19997
        sim.simxFinish(-1)  # just in case, close all opened connections
        self.clientID = sim.simxStart(ip, port, True, True, -5000, 5)
        if self.clientID == -1:
            import sys
            sys.exit('\nV-REP remote API server connection failed (' + ip + ':' + str(port) + '). Is V-REP running?')
        print('Connected to Remote API Server')
        
        with open('primitive_base/dance_primitive_library_interpole.json') as f:
            self.dance_primitive_library = json.load(f)
        
        self.Body = {}
        get_first_handles(self.clientID, self.Body)    #get first handles of Nao in the virtual environment
        self.joint_actuator = Joint_Actuator(self.clientID, self.Body)
        
        self.init_time = time.time()
        self.time_sleep = 0.01
        self.i = 0
    
    def actuate(self, queue_primitive_from_selector, queue_primitive_from_actuator, queue_frame, queue_time_sleep):
        if not queue_primitive_from_selector.empty():
            primitive = queue_primitive_from_selector.get()
            #print(primitive)
            num_frame = len(self.dance_primitive_library[str(primitive)])
            current_time = time.time()-self.init_time
            if not queue_primitive_from_actuator.empty():
                _ = queue_primitive_from_actuator.get()
            queue_primitive_from_actuator.put((current_time, num_frame, self.i))
            self.i +=1
            #print('frmae', num_frame)
            #t1=time.time()
            for frame in range(num_frame):
                #print(time.time()-t1)
                #t1=time.time()
                #time_1=time.time()
                if not queue_frame.empty():
                    _ = queue_frame.get()
                queue_frame.put(frame)
                self.joint_actuator.joint_actuate(primitive, frame)
                #time.sleep(0.005)
                if not queue_time_sleep.empty():
                    self.time_sleep = queue_time_sleep.get()
                #time_transmit = time.time()-time_1
                time.sleep(max(0.0001, self.time_sleep))