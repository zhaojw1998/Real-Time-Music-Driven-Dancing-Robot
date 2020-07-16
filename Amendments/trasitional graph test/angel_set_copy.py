import json
import numpy as np
from math import *

PI = np.pi
HIPOFFSET = 100
THIGH = 100
TIBIA = 102.90
NECK = 211.50
a=0
b=.5 #control LHR, RHR
#with open("C:\\Users\\lenovo\\Desktop\\MyNao\\motion_base\\motion_base.json",'r') as f:
#    motion = json.load(f)['9']['frame']

def set_angel(motion, frame):
    i = frame
    #for left shoulder
    z = np.array(motion[str(i)]['8']) - np.array(motion[str(i)]['7'])
    x = np.cross(np.array(motion[str(i)]['11'])-np.array(motion[str(i)]['8']), z)
    y = np.cross(z, x)
    LS = np.array(motion[str(i)]['12']) - np.array(motion[str(i)]['11'])
    LSP = np.arccos(np.dot(z, np.cross(LS, y))/(np.linalg.norm(z)*np.linalg.norm(np.cross(LS, y))))
    LSP_aux = np.arccos(np.dot(LS, z)/(np.linalg.norm(LS)*np.linalg.norm(z)))
    LSP = b * np.sign(LSP_aux-PI/2) * LSP
    LSR = PI/2 - np.arccos(np.dot(LS, y)/(np.linalg.norm(LS)*np.linalg.norm(y)))

    #for left elbow
    z = -LS
    x = np.cross(np.array(motion[str(i)]['11'])-np.array(motion[str(i)]['8']), z)
    y = np.cross(z, x)
    LE = np.array(motion[str(i)]['13']) - np.array(motion[str(i)]['12'])
    LEY = np.arccos(np.dot(x, np.cross(z, LE))/(np.linalg.norm(x)*np.linalg.norm(np.cross(z, LE))))
    LEY_aux = np.arccos(np.dot(x, LE)/(np.linalg.norm(x)*np.linalg.norm(LE)))
    LEY = -b * np.sign(LEY_aux-PI/2) * LEY
    LER =  np.arccos(np.dot(z, LE)/(np.linalg.norm(z)*np.linalg.norm(LE))) - PI

    #for right shoulder
    z = np.array(motion[str(i)]['8']) - np.array(motion[str(i)]['7'])
    x = np.cross(np.array(motion[str(i)]['8'])-np.array(motion[str(i)]['14']), z)
    y = np.cross(z, x)
    RS = np.array(motion[str(i)]['15']) - np.array(motion[str(i)]['14'])
    RSP = np.arccos(np.dot(z, np.cross(RS, y))/(np.linalg.norm(z)*np.linalg.norm(np.cross(RS, y))))
    RSP_aux = np.arccos(np.dot(RS, z)/(np.linalg.norm(RS)*np.linalg.norm(z)))
    RSP = b * np.sign(RSP_aux-PI/2) * RSP
    RSR = PI/2 - np.arccos(np.dot(RS, y)/(np.linalg.norm(RS)*np.linalg.norm(y)))

    #for right elbow
    z = -RS
    x = np.cross(np.array(motion[str(i)]['8'])-np.array(motion[str(i)]['14']), z)
    y = np.cross(z, x)
    RE = np.array(motion[str(i)]['16']) - np.array(motion[str(i)]['15'])
    REY = np.arccos(np.dot(-x, np.cross(z, RE))/(np.linalg.norm(x)*np.linalg.norm(np.cross(z, RE))))
    REY_aux = np.arccos(np.dot(x, RE)/(np.linalg.norm(x)*np.linalg.norm(RE)))
    REY = -b * np.sign(PI/2-REY_aux) * REY
    RER = PI - np.arccos(np.dot(z, RE)/(np.linalg.norm(z)*np.linalg.norm(RE)))

    # for left hip
    z = np.array(motion[str(i)]['7']) - np.array(motion[str(i)]['0'])
    x = np.cross(np.array(motion[str(i)]['4'])-np.array(motion[str(i)]['0']), z)
    y = np.cross(z, x)
    LH = np.array(motion[str(i)]['5']) - np.array(motion[str(i)]['4'])
    LHR = (np.sign(np.dot(y, LH)) * np.arccos(np.dot(y, np.cross(x, LH))/(np.linalg.norm(y)*np.linalg.norm(np.cross(x, LH)))))
    LHP = (np.arccos(np.dot(x, LH)/(np.linalg.norm(x)*np.linalg.norm(LH))) - PI/2)*0

    # for left knee
    LK = np.array(motion[str(i)]['6']) - np.array(motion[str(i)]['5'])
    LKP = np.arccos(np.dot(LH, LK)/(np.linalg.norm(LH)*np.linalg.norm(LK))) * 0.5
   
    # for right hip
    z = np.array(motion[str(i)]['7']) - np.array(motion[str(i)]['0'])
    x = np.cross(np.array(motion[str(i)]['0'])-np.array(motion[str(i)]['1']), z)
    y = np.cross(z, x)
    RH = np.array(motion[str(i)]['2']) - np.array(motion[str(i)]['1'])
    RHR = (np.sign(np.dot(y, RH)) * np.arccos(np.dot(y, np.cross(x, RH))/(np.linalg.norm(y)*np.linalg.norm(np.cross(x, RH)))))
    RHP = (np.arccos(np.dot(x, RH)/(np.linalg.norm(x)*np.linalg.norm(RH))) - PI/2)*0

    if LHR - RHR < 0:
        LHYP = -(RHR-LHR)/2*2
    else:
        LHYP = 0
    RHYP = LHYP

    # for right knee
    RK = np.array(motion[str(i)]['3']) - np.array(motion[str(i)]['2'])
    RKP = np.arccos(np.dot(RH, RK)/(np.linalg.norm(RH)*np.linalg.norm(RK))) * 0.5

    #for right ankle
    ROOT_coordinate = np.array([0, -50, 0 ])
    spin_axis = np.array([0, 1, 1])/np.linalg.norm(np.array([0, 1, 1]))
    v = np.array([[0, 0, 1, 0, 0, 1], 
                  [0, 1, 0, 1, 1, 0], 
                  [-1, 0, 0, 0, 0, 0]])
    v_1 = np.dot(rotation_matrix(spin_axis, RHYP), v)
    spin_axis = v_1[:, 2]
    v_2 = np.dot(rotation_matrix(spin_axis, RHR), v_1)
    spin_axis = v_2[:, 1]
    v_3 = np.dot(rotation_matrix(spin_axis, RHP), v_2)
    RK_coordinate = ROOT_coordinate + THIGH * v_3[:, 0]
    spin_axis = v_3[:, 3]
    v_4 = np.dot(rotation_matrix(spin_axis, RKP), v_3)
    RA_coordinate = RK_coordinate + TIBIA * v_4[:, 0]
    k_RAP = v_4[:, 4]
    k_RAR = v_4[:, 5]

    #for left ankle
    ROOT_coordinate = np.array([0, 50, 0 ])
    spin_axis = np.array([0, 1, -1])/np.linalg.norm(np.array([0, 1, -1]))
    v = np.array([[0, 0, 1, 0, 0, 1], 
                  [0, 1, 0, 1, 1, 0], 
                  [-1, 0, 0, 0, 0, 0]])
    v_1 = np.dot(rotation_matrix(spin_axis, LHYP), v)
    spin_axis = v_1[:, 2]
    v_2 = np.dot(rotation_matrix(spin_axis, LHR), v_1)
    spin_axis = v_2[:, 1]
    v_3 = np.dot(rotation_matrix(spin_axis, LHP), v_2)
    LK_coordinate = ROOT_coordinate + THIGH * v_3[:, 0]
    spin_axis = v_3[:, 3]
    v_4 = np.dot(rotation_matrix(spin_axis, LKP), v_3)
    LA_coordinate = LK_coordinate + TIBIA * v_4[:, 0]
    k_LAP = v_4[:, 4]
    k_LAR = v_4[:, 5]

    #calculation
    NK_coordinate = np.array([0, 0, 0.8*NECK])
    z = np.cross(np.cross(RA_coordinate-NK_coordinate, LA_coordinate-RA_coordinate), LA_coordinate-RA_coordinate)
    z = np.array([0, 0, 1])

    RLEG_up = np.cross(k_RAR, k_RAP)
    tmp = np.cross(k_RAP, z)
    RAP = np.arccos(np.dot(RLEG_up, tmp)/(np.linalg.norm(RLEG_up)*np.linalg.norm(tmp))) - PI/2
    RAR = PI/2 - np.arccos(np.dot(k_RAP, -z)/(np.linalg.norm(k_RAP)*np.linalg.norm(z)))

    LLEG_up = np.cross(k_LAR, k_LAP)
    tmp = np.cross(k_LAP, z)
    LAP = np.arccos(np.dot(LLEG_up, tmp)/(np.linalg.norm(LLEG_up)*np.linalg.norm(tmp))) - PI/2
    LAR = PI/2 - np.arccos(np.dot(k_LAP, -z)/(np.linalg.norm(k_LAP)*np.linalg.norm(z)))
    if LHR - RHR < 0:
        print(LHR - RHR)
    


    """
    z = np.array([0, 0, 1])
    leg = np.array(motion[str(i)]['2'])-np.array(motion[str(i)]['3'])
    k_RAP = np.cross(np.array(motion[str(i)]['2'])-np.array(motion[str(i)]['1']), np.array(motion[str(i)]['3'])-np.array(motion[str(i)]['2']))
    k_RAR =np.cross(k_RAP, np.array(motion[str(i)]['2'])-np.array(motion[str(i)]['3']))
    tmp = np.cross(k_RAP, z)
    RAP = np.arccos(np.dot(leg, z)/(np.linalg.norm(leg)*np.linalg.norm(z))) - PI/2
    RAR = PI/2 - np.arccos(np.dot(k_RAP, -z)/(np.linalg.norm(k_RAP)*np.linalg.norm(z)))
    
    #for left ankle
    z = np.array([0, 0, 1])
    leg = np.array(motion[str(i)]['5'])-np.array(motion[str(i)]['6'])
    k_LAP = np.cross(np.array(motion[str(i)]['5'])-np.array(motion[str(i)]['4']), np.array(motion[str(i)]['6'])-np.array(motion[str(i)]['5']))
    k_LAR =np.cross(k_LAP, np.array(motion[str(i)]['5'])-np.array(motion[str(i)]['6']))
    tmp = np.cross(k_LAP, z)
    LAP = np.arccos(np.dot(leg, z)/(np.linalg.norm(leg)*np.linalg.norm(z))) - PI/2
    LAR = PI/2 - np.arccos(np.dot(k_LAP, -z)/(np.linalg.norm(k_LAP)*np.linalg.norm(z)))
    """

    return LSP, LSR, LEY, LER, RSP, RSR, REY, RER, LHR, LHP, LKP, RHR, RHP, RKP, LAR, RAR, LAP, RAP, LHYP, RHYP

def rodrigues(v, k, theta):
    return torch.cos(theta)*v + (1-torch.cos(theta))*(torch.dot(v, k))*k + torch.sin(theta)*torch.cross(k, v)

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    axis is the rotation axis, which should be a unit vector
    """
    theta = np.asarray(theta)
    a = np.cos(theta/2)
    b, c, d = -axis * np.sin(theta/2)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])