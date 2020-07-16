import numpy as np
from math import *
#from manage_joints import get_first_handles
import os
#import sim
import time
from tqdm import tqdm
from scipy import interpolate
import sys

class coordinate2angle(object):
    def __init__(self):
        self.frame = 0#motion_frame
        self.spaceResolution = 100
        self.timeResolution = 100
        self.bound_range = 0
        self.step = 0

    def set_bound(self, bound_range, step, space_resolution=32, time_resolution=16):
        self.bound_range = bound_range
        self.step = step
        self.spaceResolution = space_resolution
        self.timeResolution = time_resolution
    
    def coordinate2angel_continuous(self, frame):
        """input: a motion frame in size (num_keypoints, 3)"""
        PI = np.pi
        #for left shoulder:
        z = np.array(frame[8]) - np.array(frame[7])# 8 7
        x = np.cross((np.array(frame[11])-np.array(frame[8])), z) #11 8
        y = np.cross(z, x)
        LS = np.array(frame[12]) - np.array(frame[11])
        LSP = np.arccos(np.dot(z, np.cross(LS, y))/(np.linalg.norm(z)*np.linalg.norm(np.cross(LS, y))))
        LSP_aux = np.arccos(np.dot(LS, z)/(np.linalg.norm(LS)*np.linalg.norm(z)))
        LSP = np.sign(LSP_aux-PI/2) * LSP #* 0.5
        LSR = PI/2 - np.arccos(np.dot(LS, y)/(np.linalg.norm(LS)*np.linalg.norm(y)))
        #for left elbow
        z = -LS
        x = np.cross(np.array(frame[11])-np.array(frame[8]), z)
        y = np.cross(z, x)
        LE = np.array(frame[13]) - np.array(frame[12])
        LEY = np.arccos(np.dot(x, np.cross(z, LE))/(np.linalg.norm(x)*np.linalg.norm(np.cross(z, LE))))
        LEY_aux = np.arccos(np.dot(x, LE)/(np.linalg.norm(x)*np.linalg.norm(LE)))
        LEY = np.sign(LEY_aux-PI/2) * LEY #* 0.5
        LER =  np.arccos(np.dot(z, LE)/(np.linalg.norm(z)*np.linalg.norm(LE))) - PI

        #for right shoulder
        z = np.array(frame[8]) - np.array(frame[7]) # 8 7
        x = np.cross(np.array(frame[8])-np.array(frame[14]), z) #8 14
        y = np.cross(z, x)
        RS = np.array(frame[15]) - np.array(frame[14])
        RSP = np.arccos(np.dot(z, np.cross(RS, y))/(np.linalg.norm(z)*np.linalg.norm(np.cross(RS, y))))
        RSP_aux = np.arccos(np.dot(RS, z)/(np.linalg.norm(RS)*np.linalg.norm(z)))
        RSP = np.sign(RSP_aux-PI/2) * RSP #* 0.5
        RSR = PI/2 - np.arccos(np.dot(RS, y)/(np.linalg.norm(RS)*np.linalg.norm(y)))
        #for right elbow
        z = -RS
        x = np.cross(np.array(frame[8])-np.array(frame[14]), z)
        y = np.cross(z, x)
        RE = np.array(frame[16]) - np.array(frame[15])
        REY = np.arccos(np.dot(-x, np.cross(z, RE))/(np.linalg.norm(x)*np.linalg.norm(np.cross(z, RE))))
        REY_aux = np.arccos(np.dot(x, RE)/(np.linalg.norm(x)*np.linalg.norm(RE)))
        REY = np.sign(PI/2-REY_aux) * REY #* 0.5
        RER = PI - np.arccos(np.dot(z, RE)/(np.linalg.norm(z)*np.linalg.norm(RE)))

        # for left hip
        z = np.array(frame[7]) - np.array(frame[0])
        x = np.cross(np.array(frame[4])-np.array(frame[0]), z)
        y = np.cross(z, x)
        LH = np.array(frame[5]) - np.array(frame[4])
        LHR = (np.sign(np.dot(y, LH)) * np.arccos(np.dot(y, np.cross(x, LH))/(np.linalg.norm(y)*np.linalg.norm(np.cross(x, LH))))) #*  0.6
        LHP = (np.arccos(np.dot(x, LH)/(np.linalg.norm(x)*np.linalg.norm(LH))) - PI/2) #* 0.3
        # for left knee
        LK = np.array(frame[6]) - np.array(frame[5])
        LKP = np.arccos(np.dot(LH, LK)/(np.linalg.norm(LH)*np.linalg.norm(LK))) #* 0.5
        # for left hip again
        kY = np.cross(LH, LK)
        LHY = np.sign(np.dot(kY, x))*(PI/2 - np.sign(np.dot(kY, y))*np.arccos(np.dot(np.sign(np.dot(kY, x))*x, kY)/(np.linalg.norm(x)*np.linalg.norm(kY))))
        
        # for right hip
        z = np.array(frame[7]) - np.array(frame[0])
        x = np.cross(np.array(frame[0])-np.array(frame[1]), z)
        y = np.cross(z, x)
        RH = np.array(frame[2]) - np.array(frame[1])
        RHR = (np.sign(np.dot(y, RH)) * np.arccos(np.dot(y, np.cross(x, RH))/(np.linalg.norm(y)*np.linalg.norm(np.cross(x, RH))))) #* 0.6
        RHP = (np.arccos(np.dot(x, RH)/(np.linalg.norm(x)*np.linalg.norm(RH))) - PI/2) #* 0.3
        # for right knee
        RK = np.array(frame[3]) - np.array(frame[2])
        RKP = np.arccos(np.dot(RH, RK)/(np.linalg.norm(RH)*np.linalg.norm(RK))) #* 0.5
        #for right hip again
        kY = -np.cross(RH, RK)
        RHY = np.sign(np.dot(kY, x))*(PI/2 - np.sign(np.dot(kY, -y))*np.arccos(np.dot(np.sign(np.dot(kY, x))*x, kY)/(np.linalg.norm(x)*np.linalg.norm(kY))))

        #for torso
        upper = np.array(frame[8])-np.array(frame[7])
        lower = np.array(frame[0])-np.array(frame[7])
        TP = PI - np.arccos(np.dot(upper, lower)/(np.linalg.norm(upper)*np.linalg.norm(lower)))

        return LSP, LSR, LEY, LER, RSP, RSR, REY, RER, LHP, LHR, LHY, LKP, RHP, RHR, RHY, RKP, TP

    def bound(self, library_dir, space_resolution=32, time_resolution=16):
        """input: a motion libray directory which stores a number of motion primitives in npy."""
        self.timeResolution = time_resolution
        bound_range = np.zeros((17, 2))
        bound_range[:, 0] = 100
        bound_range[:, 1] = -100
        for sub_dir_idx in tqdm(range(len(os.listdir(library_dir)))):
            sub_dir = os.listdir(library_dir)[sub_dir_idx]
            file = os.path.join(library_dir, sub_dir) + '\\dance_motion_' + str(int(sub_dir)) + '.npy'
            try:
                primitive = np.load(file).reshape((-1, 17, 3))
            except FileNotFoundError:
                print('empty file', sub_dir)
                continue 
            #print(primitive[0])      
            next_sub_dir = os.listdir(library_dir)[(sub_dir_idx+1)%len(os.listdir(library_dir))]
            next_file = os.path.join(library_dir, next_sub_dir) + '\\dance_motion_' + str(int(next_sub_dir)) + '.npy'
            try:
                next_primitive = np.load(next_file).reshape((-1, 17, 3))
            except FileNotFoundError:
                continue
            primitive = np.concatenate((primitive, next_primitive[0][np.newaxis, :, :]), axis=0)
            #print(primitive[0])
            primitive_angle = np.zeros((primitive.shape[0], 17))
            for i in range(primitive.shape[0]):
                primitive_angle[i, :] = np.array(self.coordinate2angel_continuous(self.coordinateModify(primitive[i])))
            time_axis=np.arange(0, primitive_angle.shape[0])
            f = interpolate.interp1d(time_axis, primitive_angle, axis=0, kind='cubic')
            time_new = np.arange(0, primitive_angle.shape[0]-1, float(primitive_angle.shape[0]-1)/self.timeResolution)
            primitive_interpolated = f(time_new)
            #print(primitive_interpolated[7])
            #print(primitive_interpolated.shape)
            for i in range(primitive_interpolated.shape[0]):
                for idx in range(primitive_interpolated.shape[1]):
                    if primitive_interpolated[i, idx] < bound_range[idx, 0]:
                        bound_range[idx, 0] = primitive_interpolated[i, idx]
                    if primitive_interpolated[i, idx] > bound_range[idx, 1]:
                        bound_range[idx, 1] = primitive_interpolated[i, idx]
        self.spaceResolution = space_resolution
        self.bound_range = bound_range
        self.step = (self.bound_range[:, 1]-self.bound_range[:, 0])/(self.spaceResolution-1)
        np.save('bound_range.npy', self.bound_range)
        np.save('step', self.step)

    def coordinateModify(self, primitive):
        """input: one motion primitive in (numFrame, num_joints, 3), where 3 stands for coordinates (y, -z, -x). We convert it to (x, y, z)"""
        primitive_modified = np.zeros(primitive.shape, dtype=primitive.dtype)
        primitive_modified[:, 0] = -primitive[:, 2]
        primitive_modified[:, 1] = primitive[:, 0]
        primitive_modified[:, 2] = -primitive[:, 1]
        return primitive_modified
    
    def loadBatchData_discretize(self, library_dir, time_resolution=16):
        """input: a motion libray directory which stores a number of motion primitives in npy."""
        if type(self.bound_range) == int:
            print("bound the range first!")
            sys.exit()
        #print(self.bound_range)
        dataBatch = np.empty((0, self.timeResolution, 17*self.spaceResolution))
        primitiveNames = os.listdir(library_dir)
        for idx in tqdm(range(len(primitiveNames))):
            #load angles for each frame of this primitive
            sub_dir = primitiveNames[idx]
            file = os.path.join(library_dir, sub_dir) + '\\dance_motion_' + str(int(sub_dir)) + '.npy'
            try:
                primitive = np.load(file).reshape((-1, 17, 3))
            except FileNotFoundError:
                continue
            primitiveAngles = np.zeros((primitive.shape[0]+1, 17))
            for frame in range(primitive.shape[0]):
                frameAngles = self.coordinateModify(primitive[frame])
                primitiveAngles[frame, :] = np.array(self.coordinate2angel_continuous(frameAngles))
            #load angles for the first frame of the next frame
            next_sub_dir = primitiveNames[(idx+1)%len(primitiveNames)]
            next_file = os.path.join(library_dir, next_sub_dir) + '\\dance_motion_' + str(int(next_sub_dir)) + '.npy'
            try:
                next_primitive = np.load(next_file).reshape((-1, 17, 3))
            except FileNotFoundError:
                continue
            next_frameAngles = self.coordinateModify(next_primitive[0])
            primitiveAngles[-1, :] = np.array(self.coordinate2angel_continuous(next_frameAngles))
            #interpolate
            time_axis=np.arange(0, primitiveAngles.shape[0])
            f = interpolate.interp1d(time_axis, primitiveAngles, axis=0, kind='cubic')
            time_new = np.arange(0, primitiveAngles.shape[0]-1, float(primitiveAngles.shape[0]-1)/self.timeResolution)
            primitiveAngles_interpolated = f(time_new)
            #print(primitiveAngles_interpolated[7])
            #discretize
            data_sample = np.zeros((self.timeResolution, 17*self.spaceResolution))
            for frame_idx in range(primitiveAngles_interpolated.shape[0]):
                for angle_idx in range(primitiveAngles_interpolated.shape[1]):
                    #print((primitiveAngles_interpolated[frame_idx, angle_idx]-self.bound_range[angle_idx, 0])/self.step)
                    forward_step = int(np.round((primitiveAngles_interpolated[frame_idx, angle_idx]-self.bound_range[angle_idx, 0])/self.step[angle_idx]))
                    #if forward_step < 0:
                    #   print(frame_idx, angle_idx, primitiveAngles_interpolated[frame_idx, angle_idx]-self.bound_range[angle_idx, 0])
                    assert(forward_step >= 0)
                    hotspot = forward_step + self.spaceResolution*angle_idx
                    #print(hotspot)
                    data_sample[frame_idx, hotspot] = 1
            dataBatch = np.concatenate((dataBatch, data_sample[np.newaxis, :, :]), axis=0)
        return dataBatch

    def loadFrameData_discretize(self, file_path):
        frame_sample = np.load(file_path)[0]
        frame_sample = self.coordinateModify(frame_sample)
        frame_angles = self.coordinate2angel_continuous(frame_sample)
        #print(frame_angles)
        data_sample = np.zeros((1, 17*self.spaceResolution))
        for angle_idx in range(len(frame_angles)):
            forward_step= int(np.round((frame_angles[angle_idx]-self.bound_range[angle_idx, 0])/self.step[angle_idx]))
            #print(forward_step)
            assert(forward_step >= 0)
            hotspot = forward_step + self.spaceResolution*angle_idx
            data_sample[0, hotspot] = 1
        #print(self.bound_range)
        return data_sample
    
    def frameRecon(self, frame):
        angles_recon = []
        forward_step = 0
        #print(frame.shape)
        for idx in range(0, frame.shape[0], self.spaceResolution):
            #print(idx//self.spaceResolution)
            while(frame[idx+forward_step]) != 1:
                forward_step += 1
            #print(forward_step)
            angle_recon = forward_step*self.step[idx//self.spaceResolution] + self.bound_range[idx//self.spaceResolution][0]
            angles_recon.append(angle_recon)
            forward_step = 0
        return angles_recon

    def rotation_matrix(self, axis, theta):
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

    def generateWholeJoints(self, upperJoints):
        LSP, LSR, LEY, LER, RSP, RSR, REY, RER, LHP, LHR, LHY, LKP, RHP, RHR, RHY, RKP, TP = upperJoints
        PI = np.pi
        HIPOFFSET = 100
        THIGH = 100
        TIBIA = 102.90
        NECK = 211.50
        NECKOFFSET = 126.50
        SHOULDER = 98.0
        offset = 30
        #related vectors
        shoulder_vector = np.dot(self.rotation_matrix(np.array([1, 0, 0]), LSR), np.array([1, 0, 0]))
        shoulder_vector = np.dot(self.rotation_matrix(np.array([0, 1, 0]), LSP), shoulder_vector)
        ref_pointL = np.array([0, SHOULDER, NECKOFFSET]) - shoulder_vector * offset
        shoulder_vector = np.dot(self.rotation_matrix(np.array([1, 0, 0]), RSR), np.array([1, 0, 0]))
        shoulder_vector = np.dot(self.rotation_matrix(np.array([0, 1, 0]), RSP), shoulder_vector)
        ref_pointR = np.array([0, -SHOULDER, NECKOFFSET]) - shoulder_vector * offset
        #for the couple hip joints
        if LHR - RHR < 0:
            LHYP = -(RHR-LHR)/2*2
        else:
            LHYP = 0
        RHYP = LHYP
        #for right ankle
        ROOT_coordinate = np.array([0, -50, 0 ])
        spin_axis = np.array([0, 1, 1])/np.linalg.norm(np.array([0, 1, 1]))
        v = np.array([[0, 0, 1, 0, 0, 1], 
                    [0, 1, 0, 1, 1, 0], 
                    [-1, 0, 0, 0, 0, 0]])
        v_1 = np.dot(self.rotation_matrix(spin_axis, RHYP), v)
        spin_axis = v_1[:, 2]
        v_2 = np.dot(self.rotation_matrix(spin_axis, RHR), v_1)
        spin_axis = v_2[:, 1]
        v_3 = np.dot(self.rotation_matrix(spin_axis, RHP), v_2)
        RK_coordinate = ROOT_coordinate + THIGH * v_3[:, 0]
        spin_axis = v_3[:, 3]
        v_4 = np.dot(self.rotation_matrix(spin_axis, RKP), v_3)
        RA_coordinate = RK_coordinate + TIBIA * v_4[:, 0]
        k_RAP = v_4[:, 4]
        k_RAR = v_4[:, 5]
        #for left ankle
        ROOT_coordinate = np.array([0, 50, 0 ])
        spin_axis = np.array([0, 1, -1])/np.linalg.norm(np.array([0, 1, -1]))
        v = np.array([[0, 0, 1, 0, 0, 1], 
                    [0, 1, 0, 1, 1, 0], 
                    [-1, 0, 0, 0, 0, 0]])
        v_1 = np.dot(self.rotation_matrix(spin_axis, LHYP), v)
        spin_axis = v_1[:, 2]
        v_2 = np.dot(self.rotation_matrix(spin_axis, LHR), v_1)
        spin_axis = v_2[:, 1]
        v_3 = np.dot(self.rotation_matrix(spin_axis, LHP), v_2)
        LK_coordinate = ROOT_coordinate + THIGH * v_3[:, 0]
        spin_axis = v_3[:, 3]
        v_4 = np.dot(self.rotation_matrix(spin_axis, LKP), v_3)
        LA_coordinate = LK_coordinate + TIBIA * v_4[:, 0]
        k_LAP = v_4[:, 4]
        k_LAR = v_4[:, 5]
        #calculation
        #NK_coordinate = np.array([-0.05*NECK, 0, 1*NECK])
        NK_coordinate = (ref_pointL + ref_pointR) / 2
        z = np.cross(np.cross(RA_coordinate-NK_coordinate, LA_coordinate-RA_coordinate), LA_coordinate-RA_coordinate)
        RLEG_up = np.cross(k_RAR, k_RAP)
        tmp = np.cross(k_RAP, z)
        RAP = np.arccos(np.dot(RLEG_up, tmp)/(np.linalg.norm(RLEG_up)*np.linalg.norm(tmp))) - PI/2
        RAR = PI/2 - np.arccos(np.dot(k_RAP, -z)/(np.linalg.norm(k_RAP)*np.linalg.norm(z)))
        LLEG_up = np.cross(k_LAR, k_LAP)
        tmp = np.cross(k_LAP, z)
        LAP = np.arccos(np.dot(LLEG_up, tmp)/(np.linalg.norm(LLEG_up)*np.linalg.norm(tmp))) - PI/2
        LAR = PI/2 - np.arccos(np.dot(k_LAP, -z)/(np.linalg.norm(k_LAP)*np.linalg.norm(z)))

        return LSP, LSR, LEY, LER, RSP, RSR, REY, RER, LHP, LHR, LHYP, LKP, RHP, RHR, RHYP, RKP, LAP, LAR, RAP, RAR

            
if __name__ == '__main__':

    converter = coordinate2angel()
    converter.bound('danceprimitives_new', 32)
    data=converter.loadBatchData_discretize('danceprimitives_new', 16)
    print(data.shape)
    np.save('dance_unit_data.npy', data)

    