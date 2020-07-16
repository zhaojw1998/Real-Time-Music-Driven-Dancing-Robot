# -*- coding: utf-8 -*-

import sim
import time
import math
from manage_joints import get_first_handles
import sys
from angel_set_copy import set_angel

PI = math.pi

def joint_actuate(clientID, Body, motion, frame):
    LSP, LSR, LEY, LER, RSP, RSR, REY, RER, LHR, LHP, LKP, RHR, RHP, RKP, LAR, RAR, LAP, RAP, LHYP, RHYP = set_angel(motion, frame)
    def control(joint, angel):
        sim.simxSetJointTargetPosition(clientID, Body[joint], angel, sim.simx_opmode_oneshot)
    #print(LSP, LSR, LEY, LER, RSP, RSR, REY, RER)
    sim.simxPauseCommunication(clientID,1)
    control('LShoulderPitch', LSP)
    control('LShoulderRoll', LSR)
    control('LElbowYaw', LEY)
    control('LElbowRoll', LER)
    control('RShoulderPitch', RSP)
    control('RShoulderRoll', RSR)
    control('RElbowYaw', REY)
    control('RElbowRoll', RER)
    control('LHipRoll', LHR)
    control('LHipPitch', LHP)
    control('LKneePitch', LKP)
    control('RHipRoll',RHR)
    control('RHipPitch', RHP)
    control('RKneePitch', RKP)
    control('LAnkleRoll', LAR)
    control('RAnkleRoll', RAR)
    control('LAnklePitch', LAP)
    control('RAnklePitch', RAP)
    control('RHipYawPitch', RHYP)
    control('LHipYawPitch', LHYP)
    sim.simxPauseCommunication(clientID,0)
    #time.sleep(.04)

