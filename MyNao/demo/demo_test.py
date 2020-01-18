"""
from motion_select import motion_selector
ms = motion_selector('C:\\Users\\lenovo\\Desktop\\MyNao\\motion_base\\motion_base.json')
print(ms.show())
ms.update_motion()
print(ms.show())
ms.update_motion()
print(ms.show())
"""

from speed_control import Speed_Controller
from motion_select import Motion_Selector
from multiprocessing import Process, Queue
import time
import numpy as np
import json
import os

def beat_extractor(queue_beat):
    print('Process to extract beats: %s' % os.getpid())
    time_init = time.time()
    while(True):
        time.sleep(0.5)
        queue_beat.put(time.time()-time_init)
        print('beat_time!')

def motion_selector(queue_motion_from_selector):
    print('Process to select motion: %s' % os.getpid())
    ms = Motion_Selector('MyNao\\motion_base\\motion_base.json')
    ms.transmit_motion_info(queue_motion_from_selector)
    while(True):
        if queue_motion_from_selector.empty():
            ms.update_motion()
            ms.transmit_motion_info(queue_motion_from_selector)

def speed_controller(queue_beat, queue_motion_to_controller, queue_frame, queue_time_sleep):
    print('Process to control speed: %s' % os.getpid())
    sc = Speed_Controller(alpha=0.9, motion_base_dir='MyNao\\motion_base\\motion_base.json', kp=1e-3, ki=0, kd=0)
    sc.set_time_sleep(queue_time_sleep)
    time_init = time.time()
    while(True):
        sc.control(queue_beat, queue_motion_to_controller, queue_frame)
        sc.set_time_sleep(queue_time_sleep)
        if(time.time()-time_init >= 0.5):
            sc.show()
            time_init = time.time()

def motion_actuator(queue_motion_from_selector, queue_frame, queue_motion_to_controller, queue_time_sleep):
    print('Process to actuate motion: %s' % os.getpid())
    motion_base_dir = 'MyNao\\motion_base\\motion_base.json'
    with open(motion_base_dir, 'r') as f:
        motion_base = json.load(f)
    while(True):
        if not queue_motion_from_selector.empty():
            current_motion = queue_motion_from_selector.get()
            for i in range(len(motion_base[str(current_motion)]['frame'])):
                queue_frame.put(i)
                if not queue_time_sleep.empty():
                    time.sleep(queue_time_sleep.get())
                else:
                    time.sleep(0.033)

if __name__ == '__main__':
    print('father:',  os.getpid())
    queue_beat = Queue(maxsize=1)
    queue_motion_from_selector = Queue(maxsize=1)
    queue_motion_to_controller = Queue(maxsize=1)
    queue_frame = Queue(maxsize=1)
    queue_time_sleep = Queue(maxsize=1)

    process_beat_extract = Process(target=beat_extractor, args=(queue_beat,))
    process_motion_select = Process(target=motion_selector, args=(queue_motion_from_selector,))
    process_motion_actuate = Process(target=motion_actuator, args=(queue_motion_from_selector, queue_frame, queue_motion_to_controller, queue_time_sleep))
    process_speed_controller = Process(target=speed_controller, args=(queue_beat, queue_motion_to_controller, queue_frame, queue_time_sleep))

    process_beat_extract.start()
    process_motion_select.start()
    process_motion_actuate.start()
    process_speed_controller.start()

    process_beat_extract.join()
    process_motion_select.join()
    process_motion_actuate.join()
    process_speed_controller.join()

                
