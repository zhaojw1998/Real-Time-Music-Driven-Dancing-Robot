from coordinate2angle import coordinate2angel
from manage_joints import get_first_handles
import numpy as np
import os
import time
import sim

def transmit(clientID, Body, angel):
    sim.simxSetJointTargetPosition(clientID, Body['LShoulderPitch'], angel[0], sim.simx_opmode_oneshot)
    sim.simxSetJointTargetPosition(clientID, Body['LShoulderRoll'], angel[1], sim.simx_opmode_oneshot)
    sim.simxSetJointTargetPosition(clientID, Body['LElbowYaw'], angel[2], sim.simx_opmode_oneshot)
    sim.simxSetJointTargetPosition(clientID, Body['LElbowRoll'], angel[3], sim.simx_opmode_oneshot)
    sim.simxSetJointTargetPosition(clientID, Body['RShoulderPitch'], angel[4], sim.simx_opmode_oneshot)
    sim.simxSetJointTargetPosition(clientID, Body['RShoulderRoll'], angel[5], sim.simx_opmode_oneshot)
    sim.simxSetJointTargetPosition(clientID, Body['RElbowYaw'], angel[6], sim.simx_opmode_oneshot)
    sim.simxSetJointTargetPosition(clientID, Body['RElbowRoll'], angel[7], sim.simx_opmode_oneshot)
    sim.simxSetJointTargetPosition(clientID, Body['LHipPitch'], angel[8], sim.simx_opmode_oneshot)
    sim.simxSetJointTargetPosition(clientID, Body['LHipRoll'], angel[9], sim.simx_opmode_oneshot)
    sim.simxSetJointTargetPosition(clientID, Body['LHipYawPitch'], angel[10], sim.simx_opmode_oneshot)
    sim.simxSetJointTargetPosition(clientID, Body['LKneePitch'], angel[11], sim.simx_opmode_oneshot)
    sim.simxSetJointTargetPosition(clientID, Body['RHipPitch'], angel[12], sim.simx_opmode_oneshot)
    sim.simxSetJointTargetPosition(clientID, Body['RHipRoll'], angel[13], sim.simx_opmode_oneshot)
    sim.simxSetJointTargetPosition(clientID, Body['RHipYawPitch'], angel[14], sim.simx_opmode_oneshot)
    sim.simxSetJointTargetPosition(clientID, Body['RKneePitch'], angel[15], sim.simx_opmode_oneshot)
    sim.simxSetJointTargetPosition(clientID, Body['LAnklePitch'], angel[16], sim.simx_opmode_oneshot)
    sim.simxSetJointTargetPosition(clientID, Body['LAnkleRoll'], angel[17], sim.simx_opmode_oneshot)
    sim.simxSetJointTargetPosition(clientID, Body['RAnklePitch'], angel[18], sim.simx_opmode_oneshot)
    sim.simxSetJointTargetPosition(clientID, Body['RAnkleRoll'], angel[19], sim.simx_opmode_oneshot)

if __name__ == '__main__':
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
    errorCode, acc = sim.simxGetStringSignal(clientID, 'Acceleration', sim.simx_opmode_streaming)
    #errorCode, acc = sim.simxGetStringSignal(clientID, 'Gyrometer', sim.simx_opmode_streaming)

    converter = coordinate2angel()
    converter.bound('../danceprimitives', 32)
    primitives = converter.loadBatchData_discretize('../danceprimitives', 16)#num_primitives*time_resolution*(num_joints*space_resolution)
    
    start = 7
    end = 347
    sequence_sample = np.empty(primitives[0].shape)
    for i in range(start, end):
        sequence_sample = np.concatenate((sequence_sample, primitives[i]), axis=0)
    #print(sequence_sample.shape)

    for idx_f in range(sequence_sample.shape[0]):
        angle_recon = converter.frameRecon(sequence_sample[idx_f])
        # angles: LSP, LSR, LEY, LER, RSP, RSR, REY, RER, LHP, LHR, LHYP, LKP, RHP, RHR, RHYP, RKP, LAP, LAR, RAP, RAR
        angles = converter.generateWholeJoints(angle_recon)
        assert(len(angles)==20)
        transmit(clientID, Body, angles)
        time.sleep(0.05)
        errorCode, acc = sim.simxGetStringSignal(clientID, 'Acceleration', sim.simx_opmode_buffer)
        #errorCode, acc = sim.simxGetStringSignal(clientID, 'Gyrometer', sim.simx_opmode_buffer)
        acc = sim.simxUnpackFloats(acc)
        print(acc)
        