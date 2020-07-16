from motion_select import Motion_Selector
from speed_control import Speed_Controller
from online_beat_extract import beat_extractor, beat_simulator
from motion_actuate import Motion_Actuator
from multiprocessing import Process, Queue
import time

def motionSelector_and_speedController(queue_primitive_from_selector, queue_primitive_from_actuator, queue_beat, queue_frame, queue_time_sleep):
    motion_selector = Motion_Selector()
    speed_controller = Speed_Controller(kp=1e-2, ki=0, kd=1e-3, kp2=1e-4, alpha=0.5) #kp=1e-2, ki=5e-4, kd=5e-3, kp2=1e-3 kp=5e-2
    while True:
        if queue_primitive_from_selector.empty():
            motion_selector.transmit_motion_info(queue_primitive_from_selector)
            motion_selector.transfer()
        if not queue_primitive_from_actuator.empty():
            speed_controller.control_of_speed(queue_primitive_from_actuator, queue_time_sleep)
        if not queue_beat.empty():
            speed_controller.control_of_alignment(queue_beat, queue_frame, queue_time_sleep)

def motionActuator(queue_primitive_from_selector, queue_primitive_from_actuator, queue_frame, queue_time_sleep):
    motion_actuator = Motion_Actuator()
    while True:
        motion_actuator.actuate(queue_primitive_from_selector, queue_primitive_from_actuator, queue_frame, queue_time_sleep)

if __name__ == '__main__':
    queue_beat = Queue(maxsize=1)
    queue_primitive_from_selector = Queue(maxsize=1)
    queue_primitive_from_actuator = Queue(maxsize=1)
    queue_frame = Queue(maxsize=1)
    queue_time_sleep = Queue(maxsize=1)

    #process_beat_extract = Process(target=beat_extractor, args=(queue_beat,))
    
    process_motion_actuate = Process(target=motionActuator, args=(queue_primitive_from_selector, queue_primitive_from_actuator, queue_frame, queue_time_sleep,))
    process_beat_extract = Process(target=beat_simulator, args=(queue_beat,))
    process_motionSelector_and_speedController = Process(target=motionSelector_and_speedController, args=(queue_primitive_from_selector, queue_primitive_from_actuator, queue_beat, queue_frame, queue_time_sleep,))
    
    process_motion_actuate.start()
    process_beat_extract.start()
    process_motionSelector_and_speedController.start()
   
    process_motion_actuate.join()
    process_beat_extract.join()
    process_motionSelector_and_speedController.join()
    
