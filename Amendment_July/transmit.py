from coordinate2angle import coordinate2angle
from manage_joints import get_first_handles
import numpy as np
import os
import time
import sim
from tqdm import tqdm

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
    converter = coordinate2angle()
    converter.bound('./danceprimitives_new', 32)
    #primitives = converter.loadBatchData_discretize('./danceprimitives_new', 16)#num_primitives*time_resolution*(num_joints*space_resolution)
    #np.save('primitive_data.npy', primitives)
    primitives = np.load('primitive_data.npy')
    
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
    returnCode, position=sim.simxGetObjectPosition(clientID, Body['HeadYaw'], -1, sim.simx_opmode_streaming)
    #errorCode, acc = sim.simxGetStringSignal(clientID, 'Gyrometer', sim.simx_opmode_streaming)
    acc_record_safe = []
    acc_record_fall = []
    for i in tqdm(range(primitives.shape[0])):
        sequence_sample = np.concatenate((primitives[i], primitives[(i+1)%primitives.shape[0]]), axis=0)
        for idx_f in range(sequence_sample.shape[0]):
            angle_recon = converter.frameRecon(sequence_sample[idx_f])
            # angles: LSP, LSR, LEY, LER, RSP, RSR, REY, RER, LHP, LHR, LHYP, LKP, RHP, RHR, RHYP, RKP, LAP, LAR, RAP, RAR
            angles = converter.generateWholeJoints(angle_recon)
            assert(len(angles)==20)
            transmit(clientID, Body, angles)
            time.sleep(0.03)
            errorCode, acc = sim.simxGetStringSignal(clientID, 'Acceleration', sim.simx_opmode_buffer)
            #errorCode, acc = sim.simxGetStringSignal(clientID, 'Gyrometer', sim.simx_opmode_buffer)
            acc = sim.simxUnpackFloats(acc)
            # tmp_acc.append(np.sqrt((np.square(acc[0]+)+np.square(acc[1]))/2))
            returnCode, position=sim.simxGetObjectPosition(clientID, Body['HeadYaw'], -1, sim.simx_opmode_buffer)
            if position[2] < 0.4 and position[2] > 0:   #fall down
                print(np.sqrt((np.square(acc[0])+np.square(acc[1]))/2), max(acc_record_safe))
                sim.simxStopSimulation(clientID, sim.simx_opmode_oneshot)
                time.sleep(2)
                sim.simxStartSimulation(clientID, sim.simx_opmode_oneshot)
                time.sleep(2)
                acc_record_fall.append(np.sqrt((np.square(acc[0])+np.square(acc[1]))/2))
                errorCode, acc = sim.simxGetStringSignal(clientID, 'Acceleration', sim.simx_opmode_streaming)
                returnCode, position=sim.simxGetObjectPosition(clientID, Body['HeadYaw'], -1, sim.simx_opmode_streaming)
                position = 1
                break
            acc_record_safe.append(acc)
