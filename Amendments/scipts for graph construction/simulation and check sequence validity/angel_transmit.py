# -*- coding: utf-8 -*-

import sim
import time
import math
from manage_joints import get_first_handles
import sys
from angel_set_copy import set_angel
import torch
import numpy as np 
#from model import NeuralNetwork

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

def joint_actuate(clientID, Body, motion, frame, signal):
    LSP, LSR, LEY, LER, RSP, RSR, REY, RER, LHR, LHP, LKP, RHR, RHP, RKP, LAR, RAR, LAP, RAP, LHYP, RHYP = set_angel(motion, frame, signal)

    def control(joint, angel):
        sim.simxSetJointTargetPosition(clientID, Body[joint], angel, sim.simx_opmode_oneshot)
    #print(LSP, LSR, LEY, LER, RSP, RSR, REY, RER)
    a=1
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
    control('RHipRoll', RHR)
    control('RHipPitch', RHP)
    control('RKneePitch', RKP)
    control('LAnkleRoll', LAR)
    control('RAnkleRoll', RAR)
    control('LAnklePitch', LAP)
    control('RAnklePitch', RAP)
    control('RHipYawPitch', LHYP)
    control('LHipYawPitch', RHYP)
    sim.simxPauseCommunication(clientID,0)
    #time.sleep(.04)

def actuate(clientID, Body, frame):
    LSP, LSR, LEY, LER, RSP, RSR, REY, RER, LHR, LHP, LKP, RHR, RHP, RKP, LAR, RAR, LAP, RAP, LHYP, RHYP = frame.tolist()

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
    control('RHipRoll', RHR)
    control('RHipPitch', RHP)
    control('RKneePitch', RKP)
    control('LAnkleRoll', LAR)
    control('RAnkleRoll', RAR)
    control('LAnklePitch', LAP)
    control('RAnklePitch', RAP)
    control('RHipYawPitch', LHYP)
    control('LHipYawPitch', RHYP)
    sim.simxPauseCommunication(clientID,0)
    #time.sleep(.04)





if __name__=='__main__':
    ""
    ip = '127.0.0.1'
    port = 19997
    sim.simxFinish(-1)  # just in case, close all opened connections
    clientID = sim.simxStart(ip, port, True, True, -5000, 5)
    # Connect to V-REP
    if clientID == -1:
        import sys
        sys.exit('\nV-REP remote API server connection failed (' + ip + ':' + str(port) + '). Is V-REP running?')
    print('Connected to Remote API Server')  # show in the terminal
    sim.simxStartSimulation(clientID, sim.simx_opmode_oneshot)
    Body = {}
    get_first_handles(clientID,Body)

    primitive = np.load('3DMPPE_POSENET_RELEASE/main/coord_out.npy')
    motion = {}
    for i in range(primitive.shape[0]):
        motion[str(i)] = {}
        for j in range(primitive.shape[1]):
            #motion[str(i)][str(j)] = [primitive[i][j][0], primitive[i][j][2], primitive[i][j][1]]
            motion[str(i)][str(j)] = [-primitive[i][j][2], primitive[i][j][0], -primitive[i][j][1]]
    for frame in range(primitive.shape[0]):
        joint_actuate(clientID, Body, motion, frame, 1)
        returnCode, position=sim.simxGetObjectPosition(clientID, Body['HeadYaw'], -1, sim.simx_opmode_buffer)
        time.sleep(0.03)
