import json
import numpy as np

PI = np.pi
HIPOFFSET = 100
THIGH = 100
TIBIA = 102.90
a=0
b=.5 #control LHR, RHR

with open("C:\\Users\\lenovo\\Desktop\\MyNao\\motion_base\\motion_base.json",'r') as f:
    motion = json.load(f)['2']['frame']
"""
with open('C:\\Users\\lenovo\\Desktop\\3DMPPE_POSENET_RELEASE\\motionTest.json') as f:
    motion = json.load(f)
"""
def set_angel(frame):
    i = frame
    #for left shoulder
    z = np.array(motion[str(i)]['8']) - np.array(motion[str(i)]['7'])
    x = np.cross(np.array(motion[str(i)]['11'])-np.array(motion[str(i)]['8']), z)
    y = np.cross(z, x)
    LS = np.array(motion[str(i)]['12']) - np.array(motion[str(i)]['11'])
    LSP = np.arccos(np.dot(z, np.cross(LS, y))/(np.linalg.norm(z)*np.linalg.norm(np.cross(LS, y))))
    LSP_aux = np.arccos(np.dot(LS, z)/(np.linalg.norm(LS)*np.linalg.norm(z)))
    LSP = np.sign(LSP_aux-PI/2) * LSP
    LSR = PI/2 - np.arccos(np.dot(LS, y)/(np.linalg.norm(LS)*np.linalg.norm(y)))

    #for left elbow
    z = -LS
    x = np.cross(np.array(motion[str(i)]['11'])-np.array(motion[str(i)]['8']), z)
    y = np.cross(z, x)
    LE = np.array(motion[str(i)]['13']) - np.array(motion[str(i)]['12'])
    LEY = np.arccos(np.dot(x, np.cross(z, LE))/(np.linalg.norm(x)*np.linalg.norm(np.cross(z, LE))))
    LEY_aux = np.arccos(np.dot(x, LE)/(np.linalg.norm(x)*np.linalg.norm(LE)))
    LEY = np.sign(LEY_aux-PI/2) * LEY
    LER =  np.arccos(np.dot(z, LE)/(np.linalg.norm(z)*np.linalg.norm(LE))) - PI

    #for right shoulder
    z = np.array(motion[str(i)]['8']) - np.array(motion[str(i)]['7'])
    x = np.cross(np.array(motion[str(i)]['8'])-np.array(motion[str(i)]['14']), z)
    y = np.cross(z, x)
    RS = np.array(motion[str(i)]['15']) - np.array(motion[str(i)]['14'])
    RSP = np.arccos(np.dot(z, np.cross(RS, y))/(np.linalg.norm(z)*np.linalg.norm(np.cross(RS, y))))
    RSP_aux = np.arccos(np.dot(RS, z)/(np.linalg.norm(RS)*np.linalg.norm(z)))
    RSP = np.sign(RSP_aux-PI/2) * RSP
    RSR = PI/2 - np.arccos(np.dot(RS, y)/(np.linalg.norm(RS)*np.linalg.norm(y)))

    #for right elbow
    z = -RS
    x = np.cross(np.array(motion[str(i)]['8'])-np.array(motion[str(i)]['14']), z)
    y = np.cross(z, x)
    RE = np.array(motion[str(i)]['16']) - np.array(motion[str(i)]['15'])
    REY = np.arccos(np.dot(-x, np.cross(z, RE))/(np.linalg.norm(x)*np.linalg.norm(np.cross(z, RE))))
    REY_aux = np.arccos(np.dot(x, RE)/(np.linalg.norm(x)*np.linalg.norm(RE)))
    REY = np.sign(PI/2-REY_aux) * REY
    RER = PI - np.arccos(np.dot(z, RE)/(np.linalg.norm(z)*np.linalg.norm(RE)))

    # for left hip
    z = np.array(motion[str(i)]['7']) - np.array(motion[str(i)]['0'])
    x = np.cross(np.array(motion[str(i)]['4'])-np.array(motion[str(i)]['0']), z)
    y = np.cross(z, x)
    LH = np.array(motion[str(i)]['5']) - np.array(motion[str(i)]['4'])
    LHR = np.sign(np.dot(y, LH)) * np.arccos(np.dot(y, np.cross(x, LH))/(np.linalg.norm(y)*np.linalg.norm(np.cross(x, LH)))) * b
    LHP = np.arccos(np.dot(x, LH)/(np.linalg.norm(x)*np.linalg.norm(LH))) - PI/2

    # for left knee
    LK = np.array(motion[str(i)]['6']) - np.array(motion[str(i)]['5'])
    LKP = np.arccos(np.dot(LH, LK)/(np.linalg.norm(LH)*np.linalg.norm(LK))) *a
   
    # for right hip
    z = np.array(motion[str(i)]['7']) - np.array(motion[str(i)]['0'])
    x = np.cross(np.array(motion[str(i)]['0'])-np.array(motion[str(i)]['1']), z)
    y = np.cross(z, x)
    RH = np.array(motion[str(i)]['2']) - np.array(motion[str(i)]['1'])
    RHR = np.sign(np.dot(y, RH)) * np.arccos(np.dot(y, np.cross(x, RH))/(np.linalg.norm(y)*np.linalg.norm(np.cross(x, RH)))) * b
    RHP = np.arccos(np.dot(x, RH)/(np.linalg.norm(x)*np.linalg.norm(RH))) - PI/2

    # for right knee
    RK = np.array(motion[str(i)]['3']) - np.array(motion[str(i)]['2'])
    RKP = np.arccos(np.dot(RH, RK)/(np.linalg.norm(RH)*np.linalg.norm(RK))) *a

    # for left and right ankle
    LL = THIGH * np.cos(LHP) + TIBIA * np.cos(LKP + LHP)
    RL = THIGH * np.cos(RHP) + TIBIA * np.cos(RKP + RHP)
    LL_aux = np.cos(RHR) / np.sin(abs(LHR - RHR)) * HIPOFFSET
    RL_aux = np.cos(LHR) / np.sin(abs(LHR - RHR)) * HIPOFFSET

    sign = np.sign(LHR - RHR)
    theta1 = np.arctan(np.sin(abs(LHR-RHR))/((RL_aux + sign*RL)/(LL_aux + sign*LL) - np.cos(LHR-RHR)))
    theta2 = np.arctan(np.sin(abs(LHR-RHR))/((LL_aux + sign*LL)/(RL_aux + sign*RL) - np.cos(LHR-RHR)))
    if theta1 > 0:
        RAR = sign * (PI/2 - theta1)
    else:
        RAR = sign * (-PI/2 - theta1)
    if theta2 > 0:
        LAR = sign * (-PI/2 + theta2)
    else:
        LAR = sign * (PI/2 + theta2)
    
    LAP = LHP - LKP
    RAP = LHP - RKP

    return LSP, LSR, LEY, LER, RSP, RSR, REY, RER, LHR, LHP, LKP, RHR, RHP, RKP, LAR, RAR, LAP, RAP