import json
import numpy as np
class Motion_Selector(object):
    def __init__(self, motion_base_dir):
        with open (motion_base_dir, 'r') as f:
            self.motion_base = json.load(f)
        self.current_motion = 0
        self.repeat_pb = 0.2

    def transfer(self, current_motion):
        if self.motion_base[str(current_motion)]['feature']['symmetric']:   # there exists a motion symmetric to this motion
            if current_motion % 2 == 0: # an even label refers to the second-half symmetric motion
                if ['repeat'] and np.random.rand() < self.repeat_pb:  # if repeats, jump to the first-half symmetric motion with certain probability
                    return current_motion - 1
                else:
                    while(True):
                        next_motion = np.random.randint(len(self.motion_base))
                        if next_motion % 2 == 0 and self.motion_base[str(next_motion)]['feature']['symmetric']:
                            continue    # reselect if this motion is the second-half of a symmetric motion
                        else:
                            break
                    return next_motion
            else:   # an odd label refers to the first-half symmetric motion
                return current_motion + 1   # jump to the second-half

        if self.motion_base[str(current_motion)]['feature']['repeat'] and np.random.rand() < self.repeat_pb:    # repeat if this motion tends to repeat itself
            return current_motion

        while(True):    #select randomly if not repeat and symmetric
            next_motion = np.random.randint(len(self.motion_base))
            if next_motion % 2 == 0 and self.motion_base[str(next_motion)]['feature']['symmetric']:
                continue    # reselect if this motion is the second-half of a symmetric motion
            elif next_motion == 3 or next_motion == 8 or next_motion == 6 or next_motion == 9:  #skip these motions because they are not implemented well
                continue
            else:
                break
        return next_motion

    def update_motion(self):
        self.current_motion = self.transfer(self.current_motion)

    def transmit_motion_info(self, queue):
        queue.put(self.current_motion)

    def show(self):
        return self.current_motion

