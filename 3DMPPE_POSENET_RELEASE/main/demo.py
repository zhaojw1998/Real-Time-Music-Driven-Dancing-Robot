import argparse
from tqdm import tqdm
import numpy as np
import cv2
from config import cfg
import torch
from base import Tester
from utils.vis import vis_keypoints
from utils.pose_utils import flip
import torch.backends.cudnn as cudnn

from dataset import generate_patch_image #BML Added
import torchvision.transforms as transforms

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')

    args = parser.parse_args()

    # test gpus
    if not args.gpu_ids:
        args.gpu_ids = str(np.argmin(mem_info()))

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = 0 if not gpus[0].isdigit() else int(gpus[0])
        gpus[1] = len(mem_info()) if not gpus[1].isdigit() else int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))
    
    return args

def prepare_input(imgpath, bbox):
    cvimg = cv2.imread(imgpath, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if not isinstance(cvimg, np.ndarray):
        raise IOError("Fail to read %s" % imgpath)
    img_height, img_width, img_channels = cvimg.shape

    scale, rot, do_flip, color_scale, do_occlusion = 1.0, 0.0, False, [1.0, 1.0, 1.0], False

    # 3. crop patch from img and perform data augmentation (flip, rot, color scale, synthetic occlusion)
    # Adds black padding and ensured 255x255 dimensions for input even by introducing stretching.
    img_patch, trans = generate_patch_image(cvimg, bbox, do_flip, scale, rot, do_occlusion)

    for i in range(img_channels):
        img_patch[:, :, i] = np.clip(img_patch[:, :, i] * color_scale[i], 0, 255)

    transform = transforms.Compose([ \
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std)] \
        )

    img_patch = transform(img_patch)

    input_img = img_patch.unsqueeze(0)

    return input_img

def plot_3dkeypoints(pred, skeleton):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.axes._axes import _log as matplotlib_axes_logger
    mpl.use("TKAgg", warn=False, force=True)
    matplotlib_axes_logger.setLevel('ERROR')

    def set_axes_radius(ax, origin, radius):
        ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
        ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
        ax.set_zlim3d([origin[2] - radius, origin[2] + radius])

    def set_axes_equal(ax):
        '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
        cubes as cubes, etc..  This is one possible solution to Matplotlib's
        ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

        Call this function before plt.show()

        Input
          ax: a matplotlib axis, e.g., as output from plt.gca().
        '''

        limits = np.array([
            ax.get_xlim3d(),
            ax.get_ylim3d(),
            ax.get_zlim3d(),
        ])

        origin = np.mean(limits, axis=1)
        radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
        set_axes_radius(ax, origin, radius)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(pred) + 2)]
    colors = [np.array((c[2], c[1], c[0])) for c in colors]

    for i, (x, y, z) in enumerate(pred):
        ax.scatter(x, y, z, c=np.array(colors[i]), marker='o')

    for (joint_a, joint_b) in skeleton:
        ax.plot([pred[joint_a][0], pred[joint_b][0]],
                [pred[joint_a][1], pred[joint_b][1]],
                [pred[joint_a][2], pred[joint_b][2]])

    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    ax.set_zlabel('Y Label')
    # The model by default predicts x and z to be the the width-wise and length-wise directions of the original image
    # With depth added into the y axis. So this rotation shows the original 2D perspective (i.e. the 2D keypoints)
    ax.view_init(azim=-90, elev=-90)
    ax.legend()
    set_axes_equal(ax)
    plt.show()



def main():

    args = parse_args()
    cfg.set_args(args.gpu_ids)
    cudnn.fastest = True
    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True

    tester = Tester(24)

    # The pretrained model used is assumed to be  Human36M keypoints and skeleton. Change if a different model is used.
    tester.joint_num = 18
    tester.skeleton = (
        (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16), (0, 1), (1, 2), (2, 3),
        (0, 4), (4, 5), (5, 6))
    #tester._make_batch_generator()
    ##load 3D pose estimation model
    tester._make_model()

    ##loading yolo detector
    """detector = YOLOv3( model_def="/data1/cx/project/3DMPPE_POSENET_RELEASE/common/detectors/yolo/config/yolov3.cfg",
                       class_path="/data1/cx/project/3DMPPE_POSENET_RELEASE/common/detectors/yolo/data/coco.names",
                       weights_path="/data1/cx/project/3DMPPE_POSENET_RELEASE/common/detectors/yolo/weights/yolov3.weights",
                       classes=('person',),
                       max_batch_size=16,
                       device=torch.device('cuda:{}'.format(cfg.gpu_ids[0])))"""

    preds = []
    with torch.no_grad():
        itr = 1 # iteration
        imgpath = 'D:\\Robot Dance\\AI-Project-Portfolio\\3DMPPE_POSENET_RELEASE\\image\\test.jpg'
        # imgpath = '/data1/cx/project/3DMPPE_POSENET_RELEASE/image/test.jpg'
        bbox = (0, 0, 256, 256)
        input_img = prepare_input(imgpath, bbox)
        # forward
        coord_out = tester.model(input_img)
        print('coord out shape:', coord_out.shape)

        vis = True
        if vis:
            filename = str(itr)
            tmpimg = input_img[0].cpu().numpy()
            tmpimg = tmpimg * np.array(cfg.pixel_std).reshape(3,1,1) + np.array(cfg.pixel_mean).reshape(3,1,1)
            tmpimg = tmpimg.astype(np.uint8)
            tmpimg = tmpimg[::-1, :, :]
            tmpimg = np.transpose(tmpimg,(1,2,0)).copy()
            tmpkps = np.zeros((3,tester.joint_num))
            tmpkps[:2,:] = coord_out[0,:,:2].cpu().numpy().transpose(1,0) / cfg.output_shape[0] * cfg.input_shape[0]
            tmpkps[2,:] = 1
            tmpimg = vis_keypoints(tmpimg, tmpkps, tester.skeleton)
            cv2.imwrite(filename + '_output.jpg', tmpimg)
            # cv2.imshow('img', tmpimg)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        coord_out = coord_out.cpu().numpy()

        vis_3d=True
        if vis_3d:
            pred=coord_out.squeeze() #remove first batch dimension
            plot_3dkeypoints(pred, tester.skeleton)
        preds.append(coord_out)
            
    # evaluate
    preds = np.concatenate(preds, axis=0)


if __name__ == "__main__":
    main()