# -*- coding: utf-8 -*-
import sim

#Get the Handle of only one NAO
def get_first_handles(clientID,Body):    
    print('get first handles of Nao')
    #head
    Body['HeadYaw'] = sim.simxGetObjectHandle(clientID,'HeadYaw#',sim.simx_opmode_oneshot_wait)[1]
    Body['HeadPitch'] = sim.simxGetObjectHandle(clientID,'HeadPitch#',sim.simx_opmode_oneshot_wait)[1]
    #Left Leg
    Body['LHipYawPitch'] = sim.simxGetObjectHandle(clientID,'LHipYawPitch3#',sim.simx_opmode_oneshot_wait)[1]
    Body['LHipRoll'] = sim.simxGetObjectHandle(clientID,'LHipRoll3#',sim.simx_opmode_oneshot_wait)[1]
    Body['LHipPitch'] = sim.simxGetObjectHandle(clientID,'LHipPitch3#',sim.simx_opmode_oneshot_wait)[1]
    Body['LKneePitch'] = sim.simxGetObjectHandle(clientID,'LKneePitch3#',sim.simx_opmode_oneshot_wait)[1]
    Body['LAnklePitch'] = sim.simxGetObjectHandle(clientID,'LAnklePitch3#',sim.simx_opmode_oneshot_wait)[1]
    Body['LAnkleRoll'] = sim.simxGetObjectHandle(clientID,'LAnkleRoll3#',sim.simx_opmode_oneshot_wait)[1]
    #Right Leg
    Body['RHipYawPitch'] = sim.simxGetObjectHandle(clientID,'RHipYawPitch3#',sim.simx_opmode_oneshot_wait)[1]
    Body['RHipRoll'] = sim.simxGetObjectHandle(clientID,'RHipRoll3#',sim.simx_opmode_oneshot_wait)[1]
    Body['RHipPitch'] = sim.simxGetObjectHandle(clientID,'RHipPitch3#',sim.simx_opmode_oneshot_wait)[1]
    Body['RKneePitch'] = sim.simxGetObjectHandle(clientID,'RKneePitch3#',sim.simx_opmode_oneshot_wait)[1]
    Body['RAnklePitch'] = sim.simxGetObjectHandle(clientID,'RAnklePitch3#',sim.simx_opmode_oneshot_wait)[1]
    Body['RAnkleRoll'] = sim.simxGetObjectHandle(clientID,'RAnkleRoll3#',sim.simx_opmode_oneshot_wait)[1]
    #Left Arm
    Body['LShoulderPitch'] = sim.simxGetObjectHandle(clientID,'LShoulderPitch3#',sim.simx_opmode_oneshot_wait)[1]
    Body['LShoulderRoll'] = sim.simxGetObjectHandle(clientID,'LShoulderRoll3#',sim.simx_opmode_oneshot_wait)[1]
    Body['LElbowYaw'] = sim.simxGetObjectHandle(clientID,'LElbowYaw3#',sim.simx_opmode_oneshot_wait)[1]
    Body['LElbowRoll'] = sim.simxGetObjectHandle(clientID,'LElbowRoll3#',sim.simx_opmode_oneshot_wait)[1]
    Body['LWristYaw'] = sim.simxGetObjectHandle(clientID,'LWristYaw3#',sim.simx_opmode_oneshot_wait)[1]
    #Right Arm
    Body['RShoulderPitch'] = sim.simxGetObjectHandle(clientID,'RShoulderPitch3#',sim.simx_opmode_oneshot_wait)[1]
    Body['RShoulderRoll'] = sim.simxGetObjectHandle(clientID,'RShoulderRoll3#',sim.simx_opmode_oneshot_wait)[1]
    Body['RElbowYaw'] = sim.simxGetObjectHandle(clientID,'RElbowYaw3#',sim.simx_opmode_oneshot_wait)[1]
    Body['RElbowRoll'] = sim.simxGetObjectHandle(clientID,'RElbowRoll3#',sim.simx_opmode_oneshot_wait)[1]
    Body['RWristYaw'] = sim.simxGetObjectHandle(clientID,'RWristYaw3#',sim.simx_opmode_oneshot_wait)[1]
    #print('done (ignore fingers)')
    """
    #Left fingers
    print '-> Left Fingers for NAO : ' + str(1)
    Body[24].append(sim.simxGetObjectHandle(clientID,'NAO_LThumbBase#',sim.simx_opmode_oneshot_wait)[1])
    Body['HeadYaw'] = sim.simxGetObjectHandle(clientID,'HeadYaw#',sim.simx_opmode_oneshot_wait)[1]
    Body[24].append(sim.simxGetObjectHandle(clientID,'Revolute_joint8#',sim.simx_opmode_oneshot_wait)[1])
    Body['HeadYaw'] = sim.simxGetObjectHandle(clientID,'HeadYaw#',sim.simx_opmode_oneshot_wait)[1]
    Body[24].append(sim.simxGetObjectHandle(clientID,'NAO_LLFingerBase#',sim.simx_opmode_oneshot_wait)[1])
    Body['HeadYaw'] = sim.simxGetObjectHandle(clientID,'HeadYaw#',sim.simx_opmode_oneshot_wait)[1]
    Body[24].append(sim.simxGetObjectHandle(clientID,'Revolute_joint12#',sim.simx_opmode_oneshot_wait)[1])
    Body['HeadYaw'] = sim.simxGetObjectHandle(clientID,'HeadYaw#',sim.simx_opmode_oneshot_wait)[1]
    Body[24].append(sim.simxGetObjectHandle(clientID,'Revolute_joint14#',sim.simx_opmode_oneshot_wait)[1])
    Body['HeadYaw'] = sim.simxGetObjectHandle(clientID,'HeadYaw#',sim.simx_opmode_oneshot_wait)[1]
    Body[24].append(sim.simxGetObjectHandle(clientID,'NAO_LRFinger_Base#',sim.simx_opmode_oneshot_wait)[1])
    Body['HeadYaw'] = sim.simxGetObjectHandle(clientID,'HeadYaw#',sim.simx_opmode_oneshot_wait)[1]
    Body[24].append(sim.simxGetObjectHandle(clientID,'Revolute_joint11#',sim.simx_opmode_oneshot_wait)[1])
    Body['HeadYaw'] = sim.simxGetObjectHandle(clientID,'HeadYaw#',sim.simx_opmode_oneshot_wait)[1]
    Body[24].append(sim.simxGetObjectHandle(clientID,'Revolute_joint13#',sim.simx_opmode_oneshot_wait)[1])
    Body['HeadYaw'] = sim.simxGetObjectHandle(clientID,'HeadYaw#',sim.simx_opmode_oneshot_wait)[1]
    Body[25].append(Body[24][0:8])
    #Right Fingers
    print '-> Right Fingers for NAO : ' + str(1)
    Body[26].append(sim.simxGetObjectHandle(clientID,'NAO_RThumbBase#',sim.simx_opmode_oneshot_wait)[1])
    Body[26].append(sim.simxGetObjectHandle(clientID,'Revolute_joint0#',sim.simx_opmode_oneshot_wait)[1])
    Body[26].append(sim.simxGetObjectHandle(clientID,'NAO_RLFingerBase#',sim.simx_opmode_oneshot_wait)[1])
    Body[26].append(sim.simxGetObjectHandle(clientID,'Revolute_joint5#',sim.simx_opmode_oneshot_wait)[1])
    Body[26].append(sim.simxGetObjectHandle(clientID,'Revolute_joint6#',sim.simx_opmode_oneshot_wait)[1])
    Body[26].append(sim.simxGetObjectHandle(clientID,'NAO_RRFinger_Base#',sim.simx_opmode_oneshot_wait)[1])
    Body[26].append(sim.simxGetObjectHandle(clientID,'Revolute_joint2#',sim.simx_opmode_oneshot_wait)[1])
    Body[26].append(sim.simxGetObjectHandle(clientID,'Revolute_joint3#',sim.simx_opmode_oneshot_wait)[1])
    Body[27].append(Body[26][0:8]) 
"""