# -*- coding: utf-8 -*-

import sim
import time
import math
from manage_joints import get_first_handles
import sys
from angel_set import set_angel

PI = math.pi

def joint_actuate(clientID, Body, motion, frame):
    LSP, LSR, LEY, LER, RSP, RSR, REY, RER, LHR, LHP, LKP, RHR, RHP, RKP, LAR, RAR, LAP, RAP = set_angel(motion, frame)
    def control(joint, angel):
        sim.simxSetJointTargetPosition(clientID, Body[joint], angel, sim.simx_opmode_oneshot)
    #print(LSP, LSR, LEY, LER, RSP, RSR, REY, RER)
    sim.simxPauseCommunication(clientID,1)
    control('LShoulderPitch', 0.5*LSP)
    control('LShoulderRoll', LSR)
    control('LElbowYaw', -0.5*LEY)
    control('LElbowRoll', LER)
    control('RShoulderPitch', 0.5*RSP)
    control('RShoulderRoll', RSR)
    control('RElbowYaw', -0.5*REY)
    control('RElbowRoll', RER)
    control('LHipRoll', LHR)
    control('LHipPitch', LHP)
    control('LKneePitch', LKP)
    #control('LHipRoll', RHR)
    control('RHipPitch', RHP)
    control('RKneePitch', RKP)
    control('LAnkleRoll', LAR)
    control('RAnkleRoll', RAR)
    #control('LAnklePitch', LAP)
    #control('RAnklePitch', RAP)
    control('RHipYawPitch', -1*RHP)
    control('LHipYawPitch', -1*LHP)
    sim.simxPauseCommunication(clientID,0)
    #time.sleep(.04)

