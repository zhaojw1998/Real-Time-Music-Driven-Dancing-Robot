# -*- coding: utf-8 -*-

import sim
import time
import math
from manage_joints import get_first_handles
import sys
from angel_set import set_angel

PI = math.pi

WAIT = sim.simx_opmode_oneshot_wait

def show_msg(message):
    """ send a message for printing in V-REP """
    sim.simxAddStatusbarMessage(clientID, message, WAIT)
    return

ip = '127.0.0.1'
port = 19997
sim.simxFinish(-1)  # just in case, close all opened connections
clientID = sim.simxStart(ip, port, True, True, -5000, 5)
# Connect to V-REP
if clientID == -1:
    import sys
    sys.exit('\nV-REP remote API server connection failed (' + ip + ':' +
                str(port) + '). Is V-REP running?')
print('Connected to Remote API Server')  # show in the terminal
show_msg('Python: Hello')    # show in the sim

Body = {}
get_first_handles(clientID,Body)

"""
sim.simxSetJointTargetPosition(clientID,Body[14][0], PI/2, sim.simx_opmode_streaming)
time.sleep(1)
sim.simxSetJointTargetPosition(clientID,Body[19][0], PI/2, sim.simx_opmode_streaming)
time.sleep(1)
sim.simxSetJointTargetPosition(clientID,Body[3][0], PI/6, sim.simx_opmode_streaming)
F=10
for i in range(F):
    sim.simxSetJointTargetPosition(clientID,Body[8][0], -PI/2*i/F, sim.simx_opmode_streaming)

sim.simxSetJointTargetPosition(clientID,Body[5][0], PI/6, sim.simx_opmode_streaming)
sim.simxSetJointTargetPosition(clientID,Body[6][0], -PI/6, sim.simx_opmode_streaming)
sim.simxSetJointTargetPosition(clientID,Body[11][0], PI/6, sim.simx_opmode_streaming)
sim.simxSetJointTargetPosition(clientID,Body[12][0], -PI/6, sim.simx_opmode_streaming)
"""
"""
T=.5
F=10

while(True):
    sim.simxSetJointTargetPosition(clientID,Body['LElbowYaw'], -PI/2, sim.simx_opmode_streaming)
    sim.simxSetJointTargetPosition(clientID,Body['RElbowYaw'], PI/2, sim.simx_opmode_streaming)
    for f in range(1, F+1):
        sim.simxSetJointTargetPosition(clientID,Body['LShoulderRoll'], PI/3*f/F, sim.simx_opmode_streaming)
        sim.simxSetJointTargetPosition(clientID,Body['RShoulderRoll'], -PI/3*f/F, sim.simx_opmode_streaming)
        sim.simxSetJointTargetPosition(clientID,Body['LKneePitch'], PI/6*f/F, sim.simx_opmode_streaming)
        sim.simxSetJointTargetPosition(clientID,Body['LAnklePitch'], -PI/6*f/F, sim.simx_opmode_streaming)
        sim.simxSetJointTargetPosition(clientID,Body['RKneePitch'], PI/6*f/F, sim.simx_opmode_streaming)
        sim.simxSetJointTargetPosition(clientID,Body['RAnklePitch'], -PI/6*f/F, sim.simx_opmode_streaming)
        time.sleep(T/F)
    for f in range(1, F+1):
        sim.simxSetJointTargetPosition(clientID,Body['LShoulderRoll'], PI/3-PI/3*f/F, sim.simx_opmode_streaming)
        sim.simxSetJointTargetPosition(clientID,Body['RShoulderRoll'], -PI/3+PI/3*f/F, sim.simx_opmode_streaming)
        sim.simxSetJointTargetPosition(clientID,Body['LKneePitch'], PI/6-PI/6*f/F, sim.simx_opmode_streaming)
        sim.simxSetJointTargetPosition(clientID,Body['LAnklePitch'], -PI/6+PI/6*f/F, sim.simx_opmode_streaming)
        sim.simxSetJointTargetPosition(clientID,Body['RKneePitch'], PI/6-PI/6*f/F, sim.simx_opmode_streaming)
        sim.simxSetJointTargetPosition(clientID,Body['RAnklePitch'], -PI/6+PI/6*f/F, sim.simx_opmode_streaming)
        time.sleep(T/F)

    sim.simxSetJointTargetPosition(clientID,Body['LElbowYaw'], 0, sim.simx_opmode_streaming)
    sim.simxSetJointTargetPosition(clientID,Body['RElbowYaw'], 0, sim.simx_opmode_streaming)
    for f in range(1, F+1):
        sim.simxSetJointTargetPosition(clientID,Body['LShoulderPitch'], PI/2-PI/2*f/F, sim.simx_opmode_streaming)
        sim.simxSetJointTargetPosition(clientID,Body['RShoulderPitch'], PI/2-PI/2*f/F, sim.simx_opmode_streaming)
        sim.simxSetJointTargetPosition(clientID,Body['LKneePitch'], PI/6*f/F, sim.simx_opmode_streaming)
        sim.simxSetJointTargetPosition(clientID,Body['LAnklePitch'], -PI/6*f/F, sim.simx_opmode_streaming)
        sim.simxSetJointTargetPosition(clientID,Body['RKneePitch'], PI/6*f/F, sim.simx_opmode_streaming)
        sim.simxSetJointTargetPosition(clientID,Body['RAnklePitch'], -PI/6*f/F, sim.simx_opmode_streaming)
        time.sleep(T/F)
    for f in range(1, F+1):
        sim.simxSetJointTargetPosition(clientID,Body['LShoulderPitch'], PI/2*f/F, sim.simx_opmode_streaming)
        sim.simxSetJointTargetPosition(clientID,Body['RShoulderPitch'], PI/2*f/F, sim.simx_opmode_streaming)
        sim.simxSetJointTargetPosition(clientID,Body['LKneePitch'], PI/6-PI/6*f/F, sim.simx_opmode_streaming)
        sim.simxSetJointTargetPosition(clientID,Body['LAnklePitch'], -PI/6+PI/6*f/F, sim.simx_opmode_streaming)
        sim.simxSetJointTargetPosition(clientID,Body['RKneePitch'], PI/6-PI/6*f/F, sim.simx_opmode_streaming)
        sim.simxSetJointTargetPosition(clientID,Body['RAnklePitch'], -PI/6+PI/6*f/F, sim.simx_opmode_streaming)
        time.sleep(T/F)
"""
def control(joint, angel):
    sim.simxSetJointTargetPosition(clientID,Body[joint], angel, sim.simx_opmode_oneshot)

for i in range(0, 300, 1):
    LSP, LSR, LEY, LER, RSP, RSR, REY, RER, LHR, LHP, LKP, RHR, RHP, RKP, LAR, RAR, LAP, RAP = set_angel(i)
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
    time.sleep(.03)

#control('LElbowYaw', 0)
#control('LElbowRoll', -PI/2)
