# -*- coding: utf-8 -*-

import sim
import time
import math
from manage_joints import get_first_handles
import sys
from angel_set import set_angel

PI = math.pi

def init_joint(clientID, Body, motion, frame):
    def control(joint, angel):
        sim.simxSetJointTargetPosition(clientID, Body[joint], angel, sim.simx_opmode_oneshot)
    #print(LSP, LSR, LEY, LER, RSP, RSR, REY, RER)
    sim.simxPauseCommunication(clientID,1)
    control('LShoulderPitch', 0)
    control('LShoulderRoll', 0)
    control('LElbowYaw', 0)
    control('LElbowRoll', 0)
    control('RShoulderPitch', 0)
    control('RShoulderRoll', 0)
    control('RElbowYaw', 0)
    control('RElbowRoll', 0)
    control('LHipRoll', 0)
    control('LHipPitch', 0)
    control('LKneePitch', 0)
    #control('LHipRoll', 0)
    control('RHipPitch', 0)
    control('RKneePitch', 0)
    control('LAnkleRoll', 0)
    control('RAnkleRoll', 0)
    #control('LAnklePitch', 0)
    #control('RAnklePitch', 0)
    control('RHipYawPitch', 0)
    control('LHipYawPitch', 0)
    sim.simxPauseCommunication(clientID,0)
    #time.sleep(.04)

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
    #zcontrol('LAnklePitch', LAP)
    #control('RAnklePitch', RAP)
    control('RHipYawPitch', -1*RHP)
    control('LHipYawPitch', -1*LHP)
    sim.simxPauseCommunication(clientID,0)
    #time.sleep(.04)