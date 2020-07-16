# -*- coding: utf-8 -*-
import sim
import time
import json
import sys

class Joint_Actuator(object):
    def __init__(self, clientID, Body):
        with open('primitive_base/dance_primitive_library_interpole.json') as f:
            self.dance_primitive_library = json.load(f)
        self.clientID = clientID
        self.Body = Body

    def control(self, joint, angel):
        sim.simxSetJointTargetPosition(self.clientID, self.Body[joint], angel, sim.simx_opmode_oneshot)

    def joint_actuate(self, primitive, frame):
        LSP, LSR, LEY, LER, RSP, RSR, REY, RER, LHR, LHP, LKP, RHR, RHP, RKP, LAR, RAR, LAP, RAP, LHYP, RHYP = self.dance_primitive_library[str(primitive)][str(frame)]["angle_info"]
        sim.simxPauseCommunication(self.clientID,1)
        self.control('LShoulderPitch', LSP)
        self.control('LShoulderRoll', LSR)
        self.control('LElbowYaw', LEY)
        self.control('LElbowRoll', LER)
        self.control('RShoulderPitch', RSP)
        self.control('RShoulderRoll', RSR)
        self.control('RElbowYaw', REY)
        self.control('RElbowRoll', RER)
        self.control('LHipRoll', LHR)
        self.control('LHipPitch', LHP)
        self.control('LKneePitch', LKP)
        self.control('RHipRoll', RHR)
        self.control('RHipPitch', RHP)
        self.control('RKneePitch', RKP)
        self.control('LAnkleRoll', LAR)
        self.control('RAnkleRoll', RAR)
        self.control('LAnklePitch', LAP)
        self.control('RAnklePitch', RAP)
        self.control('RHipYawPitch', LHYP)
        self.control('LHipYawPitch', RHYP)
        sim.simxPauseCommunication(self.clientID,0)
        #time.sleep(.04)