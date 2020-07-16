import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class RodriguesLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def rodrigues(self, v, k, theta):
        return torch.cos(theta)*v + (1-torch.cos(theta))*(torch.dot(v, k))*k + torch.sin(theta)*torch.cross(k, v)

    def normalized(self, v):
        return v / torch.norm(v)
        
    def forward(self, angles, coordination): # angles: n × 22; coord_World: n × 51
        assert(angles.shape[0] == coordination.shape[0])
        batch_size = coordination.shape[0]
        #coordination.view(batch_size, -1, 3) # n × 17 × 3 for 3D coordinates of 17 key points
        total_loss = 0
        for i in range(batch_size):
            """transfer world coordination to torso coordination"""
            coord_W = coordination[i]
            coord_W = coord_W.view(-1, 3)
            #print(coord_W.shape)
            torso_points = torch.stack((coord_W[0],coord_W[1],coord_W[4],coord_W[7],coord_W[8],coord_W[11],coord_W[14]), dim=0) # 7 × 3
            #print(torso_points.shape)
            Y = torso_points[:, 2]
            X = torch.stack((torso_points[:, 0], torso_points[:, 1], torch.from_numpy(np.ones(torso_points.shape[0])).float()), dim = 1)
            xt = torch.matmul(torch.matmul(torch.inverse(torch.matmul(X.t(), X)), X.t()), Y)
            xt[-1] = torch.tensor(-1)
            yt = torch.cross((coord_W[8] - coord_W[0]), xt)
            zt = torch.cross(xt, yt)
            xt = self.normalized(xt)
            yt = self.normalized(yt)
            zt = self.normalized(zt)
            R_TW = torch.stack((xt, yt, zt), dim=0)     # coord_Torso = R_TW · coord_World
            coord_T = torch.matmul(coord_W, R_TW.t())
            x_T = torch.tensor([1.,0.,0.])
            y_T = torch.tensor([0.,1.,0.])
            z_T = torch.tensor([0.,0.,1.])
            """compute loss components"""   # Rodrigues: V_rot = cos(theta)*V + (1-cos(theta))*(V·k)*k + sin(theta)*(k × V)
            angle = angles[i]
            # Head
            gt_0 = self.normalized(coord_T[9]-coord_T[8])
            gt_1 = self.normalized(torch.cross(gt_0, torch.cross(coord_T[10]-coord_T[9], -gt_0)))
            v_HP = self.rodrigues(z_T, y_T, angle[0]) #*
            v_HY_pre = self.rodrigues(x_T, y_T, angle[0])
            v_HY = self.rodrigues(v_HY_pre, v_HP, angle[1]) #*
            # Left arm
            gt_2 = self.normalized(coord_T[12]-coord_T[11])
            gt_3 = self.normalized(coord_T[13]-coord_T[12])
            v_LSP = self.rodrigues(x_T, y_T, angle[3])
            v_LSR = self.rodrigues(v_LSP, self.normalized(torch.cross(coord_T[8]-coord_T[11], v_LSP)), angle[4]) #*
            v_LER = self.rodrigues(v_LSR, self.normalized(torch.cross(coord_T[8]-coord_T[11], v_LSR)), angle[5])
            v_LEY = self.rodrigues(v_LER, v_LSR, angle[6]) #*
            #Right arm
            gt_4 = self.normalized(coord_T[15]-coord_T[14])
            gt_5 = self.normalized(coord_T[16]-coord_T[15])
            v_RSP = self.rodrigues(x_T, y_T, angle[6])
            v_RSR = self.rodrigues(v_RSP, self.normalized(torch.cross(v_RSP, coord_T[8]-coord_T[14])), angle[7]) #*
            v_RER = self.rodrigues(v_RSR, self.normalized(torch.cross(v_RSR, coord_T[8]-coord_T[14])), angle[8])
            v_REY = self.rodrigues(v_RER, v_RSR, angle[9]) #*
            #Left leg
            gt_6 = self.normalized(coord_T[5]-coord_T[4])
            gt_7 = self.normalized(coord_T[6]-coord_T[5])
            v_LHP = self.rodrigues(-z_T, y_T, angle[10])
            k_LHR = self.rodrigues(x_T, y_T, angle[10])
            k_LHYP = self.rodrigues(self.normalized(torch.tensor([0.,1.,-1.])), y_T, angle[10])
            v_LHR = self.rodrigues(v_LHP, k_LHR, angle[11])
            k_LHYP = self.rodrigues(k_LHYP, k_LHR, angle[11])
            v_LHYP = self.rodrigues(v_LHR, k_LHYP, angle[12]) #*
            v_LKP = self.rodrigues(v_LHYP, self.normalized(torch.cross(coord_T[6]-coord_T[5], -v_LHYP)), angle[13]) #*
            #Right leg
            gt_8 = self.normalized(coord_T[2]-coord_T[1])
            gt_9 = self.normalized(coord_T[3]-coord_T[2])
            v_RHP = self.rodrigues(-z_T, y_T, angle[14])
            k_RHR = self.rodrigues(x_T, y_T, angle[14])
            k_RHYP = self.rodrigues(self.normalized(torch.tensor([0.,1.,1.])), y_T, angle[14])
            v_RHR = self.rodrigues(v_RHP, k_RHR, angle[15])
            k_RHYP = self.rodrigues(k_RHYP, k_RHR, angle[15])
            v_RHYP = self.rodrigues(v_RHR, k_RHYP, angle[16]) #*
            v_RKP = self.rodrigues(v_RHYP, self.normalized(torch.cross(coord_T[3]-coord_T[2], -v_RHYP)), angle[17]) #*
            #Ankles
            gt_rest = torch.tensor([0.,0.,0.])
            k_LAP = self.normalized(torch.cross(coord_T[6]-coord_T[5], coord_T[4]-coord_T[5]))
            k_LAR = self.normalized(torch.cross(k_LAP, coord_T[5]-coord_T[6]))
            k_RAP = self.normalized(torch.cross(coord_T[3]-coord_T[2], coord_T[1]-coord_T[2]))
            k_RAR = self.normalized(torch.cross(k_RAP, coord_T[2]-coord_T[3]))
            k_LAP_W = torch.matmul(k_LAP, R_TW)
            k_LAR_W = torch.matmul(k_LAR, R_TW)
            k_RAP_W = torch.matmul(k_RAP, R_TW)
            k_RAR_W = torch.matmul(k_RAR, R_TW)
            v_LAP = self.rodrigues(k_LAR_W, k_LAP_W, angle[18])
            v_LAR = self.rodrigues(k_LAP_W, v_LAP, angle[19])
            v_RAP = self.rodrigues(k_RAR_W, k_RAP_W, angle[20])
            v_RAR = self.rodrigues(k_RAP_W, v_RAP, angle[21])
            v_LAP = v_LAP * z_T #*
            v_LAR = v_LAR * z_T #*
            v_RAP = v_RAP * z_T #*
            v_RAR = v_RAR * z_T #*
            """compute loss"""
            gt = torch.stack((gt_0, gt_1, gt_2, gt_3, gt_4, gt_5, gt_6, gt_7, gt_8, gt_9, gt_rest, gt_rest, gt_rest, gt_rest), dim=0)
            sim = torch.stack((v_HP, v_HY, v_LSR, v_LEY, v_RSR, v_REY, v_LHYP, v_LKP, v_RHYP, v_RKP, v_LAP, v_LAR, v_RAP, v_RAR), dim=0)
            loss = torch.mean(torch.pow(gt-sim, 2))
            total_loss += loss

        return total_loss / batch_size