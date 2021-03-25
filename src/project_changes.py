"""Project changes to 2D images for evaluation
"""
import argparse
import os
import pickle
from collections import namedtuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from skimage.io import imread, imsave
from tqdm import tqdm

from quaternion import Quaternion


COLOR_NEG = (255, 0, 0)  # Negative: Only in run1 -> red
COLOR_POS = (0, 0, 255)  # Positive: Only in run2 -> blue
CAM_MAP = {'L': 'Left', 'R': 'Right'}


parser = argparse.ArgumentParser()
parser.add_argument('--eval_path', type=str, default='../eval_data',
                    help='Folder containing images to be evaluated')
parser.add_argument('--result_path', type=str, default='../results3d',
                    help='Folder containing the detected 3D changes')
parser.add_argument('--output_path', type=str, default='../results2d/ours',
                    help='Output folder')
parser.add_argument('--change_size', type=int, default=20,
                    help='Radius of region around each change point to mark as change')
parser.add_argument('--proj_dist', type=float, default=100.0,
                    help='Changes within this distance from the camera will be projected')
opt = parser.parse_args()


ImageInfo = namedtuple('ImageInfo', ['q_cw', 't_cw', 'intrinsics', 'img_wh'])


def get_eval_list(folder):
    """Returns the list of images to evaluate"""
    t0_folder = os.path.join(folder, 't0')
    t1_folder = os.path.join(folder, 't1')
    gt_folder = os.path.join(folder, 'groundtruth')

    t0_paths = sorted(os.listdir(t0_folder))
    t1_paths = sorted(os.listdir(t1_folder))
    gt_paths = sorted(os.listdir(gt_folder))
    eval_list = list(zip(t0_paths, t1_paths, gt_paths))
    return eval_list


def load_results(folder):
    change_pcd = o3d.io.read_point_cloud(os.path.join(folder, 'changes.pcd'))
    response = np.fromfile(os.path.join(folder, 'response.bin'), dtype=np.float32)
    with open(os.path.join(folder, 'ref_cameras.pickle'), 'rb') as fid:
        ref_cameras = pickle.load(fid)

    xyzi = np.concatenate([np.asarray(change_pcd.points), response[:, None]],
                          axis=1)
    return xyzi, ref_cameras


def project_changes(changes, cam_param, proj_dist):
    """Project changes to image plane

    Returns:
        xyd: xy, depth of change points
        response: Change response
    """

    q_cw = Quaternion(cam_param.q_cw)
    t_cw = cam_param.t_cw

    cam_xyz = -q_cw.inverse.rotate(t_cw)
    nearby_mask = np.linalg.norm(changes[:, :3] - cam_xyz[None, :], axis=1) < proj_dist
    nearby_xyzi = changes[nearby_mask, :]

    if nearby_xyzi.shape[0] == 0:
        return np.zeros((0, 3)), np.zeros(0)

    img_width, img_height = cam_param.img_wh
    fx, fy, cx, cy, k1, k2, p1, p2 = cam_param.intrinsics
    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]])
    dist_coeffs = np.array([k1, k2, p1, p2])

    pixel_coors = cv2.projectPoints(nearby_xyzi[None, :, :3],
                                    q_cw.to_axis_angle(),
                                    t_cw,
                                    K,
                                    dist_coeffs)[0]
    pixel_coors = pixel_coors[:, 0, :]
    in_image = np.logical_and(
        np.logical_and(pixel_coors[:, 0] >= 0, pixel_coors[:, 0] <= img_width),
        np.logical_and(pixel_coors[:, 1] >= 0, pixel_coors[:, 1] <= img_height))

    transformed = q_cw.rotate(nearby_xyzi[:, :3]) + t_cw[None, :]
    depths = transformed[:, 2]
    in_front = depths > 0

    mask = np.logical_and(in_image, in_front)
    xyd = np.concatenate([pixel_coors[mask, :], depths[mask, None]], axis=-1)
    response = nearby_xyzi[mask, 3]
    return xyd, response


def generate_change_image(change_px_coor, change_response, img_wh, rad):
    width, height = img_wh

    change_map_circles = np.zeros((height, width, 3), np.uint8)

    # Sort by depths from furthest to closest
    sort_idx = np.argsort(-change_px_coor[:, 2])
    change_px_coor = change_px_coor[sort_idx, :]
    change_response = change_response[sort_idx]

    for xyd, r in zip(change_px_coor, change_response):
        xy, d = xyd[:2], xyd[2]
        xy = tuple(np.round(xy).astype(np.int64))

        if r > 0:
            cv2.circle(change_map_circles, xy, rad, COLOR_POS, thickness=-1)
        else:
            cv2.circle(change_map_circles, xy, rad, COLOR_NEG, thickness=-1)

    return change_map_circles


def main():
    os.makedirs(opt.output_path, exist_ok=True)

    # Read the list of files to evaluate
    scenes = list(filter(lambda s: os.path.isdir(os.path.join(opt.eval_path, s)),
                         os.listdir(opt.eval_path)))
    for scene in scenes:
        print('Projecting changes for {} dataset'.format(scene))
        scene_folder = os.path.join(opt.eval_path, scene)
        out_folder = os.path.join(opt.output_path, scene)
        os.makedirs(out_folder, exist_ok=True)
        eval_list = get_eval_list(scene_folder)

        # Loads the detected changes
        result_folder = os.path.join(os.path.join(opt.result_path, scene))
        changes, ref_cameras = load_results(result_folder)

        for t0, t1, gt in tqdm(eval_list):

            # Retrieves the pose of im1
            _, cam, frame = t1.split('_')
            im1_key = CAM_MAP[cam] + '/' + frame
            im1_param = ref_cameras[im1_key]

            # Project the changes onto im1, and generate the change map
            change_px_coor, change_response = project_changes(changes, im1_param, opt.proj_dist)
            change_im = generate_change_image(change_px_coor, change_response, im1_param.img_wh, rad=opt.change_size)

            # Writes out the changes
            out_fname = os.path.join(opt.output_path, scene, gt)
            imsave(out_fname, change_im, check_contrast=False)


if __name__ == '__main__':
    main()