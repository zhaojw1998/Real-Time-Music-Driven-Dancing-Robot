import json
import numpy as np

class Speed_Controller(object): #for multi-core cpu (multi > 4)
    def __init__(self, alpha, motion_base_dir, kp, ki, kd):
        self.alpha = alpha
        self.spb_avg = 0
        self.beat_record = 0
        self.current_motion = 0
        self.time_sleep = 0.03
        self.d_record = 0
        self.i_history = 0
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.offset = 0.0
        with open (motion_base_dir, 'r') as f:
            self.motion_base = json.load(f)

    def control(self, queue_beat, queue_motion, queue_frame):
        if not queue_motion.empty():
            if self.spb_avg == 0:
                self.time_sleep == 0.033 #approximately equal to spb for a videos with frame_rate = 30
            else:
                self.current_motion = queue_motion.get()
                fpb = self.motion_base[str(self.current_motion)]['feature']['fpb']
                self.time_sleep = self.spb_avg / fpb    # time_sleep between two motion frames, this may not be accurate
        if not queue_beat.empty():
            current_beat = queue_beat.get()
            self.spb_avg = self.alpha*current_beat + (1-self.alpha)*self.beat_record    #running average
            self.beat_record = current_beat
            self.control_backward(queue_frame)

    def control_backward(self, queue_frame):
        if not queue_frame.empty():
            current_frame = queue_frame.get()
            last_beat = self.motion_base[str(self.current_motion)]['frame'][str(current_frame)]['last_beat']
            next_beat = self.motion_base[str(self.current_motion)]['frame'][str(current_frame)]['next_beat']
            if last_beat == 0:
                assert(last_beat == next_beat)
            else:
                if next_beat > last_beat:
                    error = last_beat
                else:
                    error = -next_beat
                self.d_record = error - self.d_record
                self.i_history += error #this might not be reasonable, only recording a last period of history is enough
                self.offset += self.kp*error + self.ki*self.i_history + self.kd*self.d_record    #PID control

    def set_time_sleep(self, queue_time_sleep):
        queue_time_sleep.put(self.time_sleep+self.offset)

    def show(self):
        print('current time_sleep:', self.time_sleep)

class Speed_Controller_and_Motion_Actuator(object): #for four-core cpu
    def __init__(self, alpha, motion_base, kp, ki, kd):
        self.alpha = alpha
        self.spb_avg = 0
        self.spb_record = 0
        self.beat_record = 0
        self.current_motion = 0
        self.current_frame = 0
        self.time_sleep = 0.033
        self.d_record = 0
        self.i_history = np.zeros(8)
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.offset = 0.0
        self.motion_base = motion_base
        self.error_record = []

    def control(self, queue_beat):
        if self.spb_avg == 0:
            self.time_sleep == 0.033 #approximately equal to spb for a videos with frame_rate = 30
        else:
            fpb = self.motion_base[str(self.current_motion)]['feature']['fpb']
            self.time_sleep = self.spb_avg / fpb    # time_sleep between two motion frames, this may not be accurate
        if not queue_beat.empty():
            current_beat = queue_beat.get()
            current_spb = current_beat - self.beat_record
            self.spb_avg = self.alpha*current_spb + (1-self.alpha)*self.spb_record    #running average
            self.beat_record = current_beat
            self.spb_record = current_spb
            #print('current beat:', current_spb)
            self.control_backward()

    def control_backward(self):
        last_beat = self.motion_base[str(self.current_motion)]['frame'][str(self.current_frame)]['last_beat']
        next_beat = self.motion_base[str(self.current_motion)]['frame'][str(self.current_frame)]['next_beat']
        if last_beat == 0:
            assert(last_beat == next_beat)
            print('error with real beat:', 0)
            self.error_record.append('0\n')
        else:
            if next_beat > last_beat:
                error = last_beat
            else:
                error = -next_beat
            self.d_record = error - self.i_history[-1]
            for i in range(self.i_history.shape[0]-1):
                self.i_history[i] = self.i_history[i+1]
            self.i_history[-1] = error #this might not be reasonable, only recording a last period of history is enough
            self.offset += (self.kp*error + self.ki*np.sum(self.i_history) + self.kd*self.d_record)    #PID control
            print('error with real beat:', error)#, 'offset:', round(self.offset, 4), 'total time_sleep:', round(self.time_sleep + self.offset, 4))
            self.error_record.append(str(error)+'\n')
        with open('MyNao\\error_record\\error_record.txt', 'w') as f:
            f.writelines(self.error_record)

    def set_current_frame(self, current_frame):
        self.current_frame = current_frame

    def set_current_motion(self, current_motion):
        self.current_motion = current_motion
    
    def get_time_sleep(self):
        return self.time_sleep + self.offset

    #def show(self):
        #print('current time_sleep:', self.time_sleep+self.offset, '({:.2}+{:.2})'.format(self.time_sleep, self.offset))
        #print('({:.2})'.format(self.offset))
