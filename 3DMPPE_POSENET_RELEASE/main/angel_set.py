import json
import numpy as np

PI = np.pi

with open("/media/jwzhao/F660DF2660DEEBFD/Users/lenovo/Desktop/3DMPPE_POSENET_RELEASE/motionTest.json",'r') as f:
    motion = json.load(f)

i = 0

#for left shoulder
z = np.array(motion[str(i)]['8']) - np.array(motion[str(i)]['7'])
x = np.cross(np.array(motion[str(i)]['11'])-np.array(motion[str(i)]['8']), z)
y = np.cross(z, x)
LS = np.array(motion[str(i)]['12']) - np.array(motion[str(i)]['11'])
LSP = np.arccos(np.dot(z, np.cross(LS, y))/(np.linalg.norm(z)*np.linalg.norm(np.cross(LS, y))))
LSP_aux = np.arccos(np.dot(LS, y)/(np.linalg.norm(LS)*np.linalg.norm(y)))
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
LER = PI - np.arccos(np.dot(z, LE)/(np.linalg.norm(z)*np.linalg.norm(LE)))

