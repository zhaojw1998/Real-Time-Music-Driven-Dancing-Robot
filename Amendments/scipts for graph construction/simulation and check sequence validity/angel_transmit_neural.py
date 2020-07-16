# -*- coding: utf-8 -*-

import sim
import time
import math
from manage_joints import get_first_handles
import sys
from angel_set import set_angel
import torch
from model import NeuralNetwork

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

def joint_actuate(clientID, Body, primitive, frame):
    model = NeuralNetwork()
    model.load_state_dict(torch.load('C:\\Users\\lenovo\\Desktop\\AI-Project-Portfolio\\Amendments\\new_angle_mapping\\weights_2.pth'))
    model.eval()
    angle = model(torch.from_numpy(primitive[frame].reshape(1, -1)).float())
    HP, HY, LSP, LSR, LER, LEY, RSP, RSR, RER, REY, LHP, LHR, LHYP, LKP, RHP, RHR, RHYP, RKP, LAP, LAR, RAP, RAR = angle.detach().numpy().reshape(-1)
    print(LSP, LSR, LER, LEY)
    def control(joint, angel):
        sim.simxSetJointTargetPosition(clientID, Body[joint], angel, sim.simx_opmode_oneshot)
    sim.simxPauseCommunication(clientID,1)
    control('HeadPitch', HP)
    control('HeadYaw', HY)
    control('LShoulderPitch', 1*LSP)
    control('LShoulderRoll', 1*LSR)
    control('LElbowRoll', 1*LER)
    control('LElbowYaw', 1*LEY)
    control('RShoulderPitch', RSP)
    control('RShoulderRoll', RSR)
    control('RElbowRoll', RER)
    control('RElbowYaw', REY)
    control('LHipPitch', LHP)
    control('LHipRoll', LHR)
    control('LHipYawPitch', LHYP)
    control('LKneePitch', LKP)
    control('RHipPitch', RHP)
    control('RHipRoll', RHR)
    control('RHipYawPitch', RHYP)
    control('RKneePitch', RKP)
    control('LAnklePitch', LAP)
    control('LAnkleRoll', LAR)
    control('RAnklePitch', RAP)
    control('RAnkleRoll', RAR)
    sim.simxPauseCommunication(clientID,0)
    #time.sleep(.04)