import time
from matplotlib import pyplot as plt
import numpy as np
import pickle

class Speed_Controller(object):
    def __init__(self, kp, ki, kd, kp2, alpha=0.5):
        self.last_beat = 0
        self.spb = 0    #second per beat
        self.alpha = alpha
        self.last_primitive = 0
        self.spp = 0    #second per primitive
        self.last_num_frame = 16
        self.last_alignment = 0
        self.time_sleep = 0.01
        self.last_error = 0
        self.sum_error = 0
        self.speed_offset = 0
        self.frame_offset = 0
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.kp2 = kp2
        self.error_record = []
        self.alignment_record = []


    def update_beat(self, new_beat):
        spb = new_beat - self.last_beat
        self.spb = self.alpha*spb + (1-self.alpha)*self.spb
        self.last_beat = new_beat
    
    def update_primitive(self, new_primitive):
        self.spp = new_primitive - self.last_primitive
        self.last_primitive = new_primitive

    def control_of_speed(self, queue_primitive, queue_time_sleep):
        #t1=time.time()
        if self.spb == 0:
            self.time_sleep = 0.033 #approximately equal to spb for a videos with frame_rate = 30
            self.speed_offset = 0
        else:
            new_primitive, num_frame, i = queue_primitive.get() #(new_primitive, num_frame)
            #print('get frame:', i)
            self.update_primitive(new_primitive)
            self.speed_feedback()
            self.time_sleep = self.time_sleep * self.last_num_frame / num_frame
            self.last_num_frame = num_frame
        if not queue_time_sleep.empty():
            _ = queue_time_sleep.get()
        queue_time_sleep.put(self.time_sleep - self.speed_offset)
        #print(self.time_sleep, '\t', self.speed_offset)
        #print('time:', time.time()-t1)
        
    def control_of_alignment(self, queue_beat, queue_frame, queue_time_sleep):
        #t1=time.time()
        current_beat = queue_beat.get()
        #print('get beat:', i)
        #print('time:', time.time()-t1)
        self.update_beat(current_beat)
        self.frame_feedback(queue_frame)
        if not queue_time_sleep.empty():
            _ = queue_time_sleep.get()
        queue_time_sleep.put(self.time_sleep - self.speed_offset - self.frame_offset)
        
        
    def speed_feedback(self):
        #print(self.spp, self.spb)
        error = self.spp + (self.last_num_frame-self.last_alignment)*self.frame_offset - self.spb
        self.error_record.append(error)
        self.sum_error += error
        self.speed_offset += self.kp*error + self.ki*self.sum_error + self.kd*(error-self.last_error)
        self.last_error = error
        #print(error)
        self.error_record.append(error)
        #if len(self.error_record) == 200:
        #    x = np.arange(0, 200)
        #    #plt.axis([0, 200, -0.01, 0.01])
        #    plt.plot(x, self.error_record, color="r", linestyle="-", linewidth=1)
        #    plt.show()


    
    def frame_feedback(self, queue_frame):
        if not queue_frame.empty():
            current_frame = queue_frame.get()
            if current_frame < self.last_num_frame-current_frame:
                error = -current_frame
            else:
                error = (self.last_num_frame-current_frame)
            #print(error)
            d_error = current_frame - self.last_alignment
            self.last_alignment = current_frame
            #if abs(error) >= 10:
            #    self.frame_offset = self.kp2*error/10
            #else:

            self.frame_offset = self.kp2*error #+ 1e-5*d_error
            
            self.alignment_record.append(error)
            if len(self.alignment_record) == 200:
                #with open('C:/Users/lenovo/Desktop/AI-Project-Portfolio/Amendments/experiment/buzhen/frame_interpole.txt', 'wb') as f:
                #    pickle.dump(self.alignment_record, f, protocol=2)
                x = np.arange(0,200)
                #plt.axis([0, 200, -0.01, 0.01])
                plt.plot(x, self.alignment_record, color="r", linestyle="-", linewidth=1)
                plt.show()

    #def set_time_sleep(self, queue_time_sleep):
    #    queue_time_sleep.put(self.time_sleep - self.speed_offset - self.frame_offset)

    def show(self):
        print('current time_sleep:', self.time_sleep)
