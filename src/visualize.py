"""This simple script visualizes the image pairs and changes.

Blue denotes disappearance, red denotes appearance of object
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--eval_path', type=str, default='../eval_data',
                    help='Folder containing images to be evaluated')
opt = parser.parse_args()
last_key = None

print('Simple visualizer for visualizing change annotations')
print('----------------------------------------------------')
print("Press 'n' to go to the next image, 'q' to quit")
print()

def on_press(event):
    global last_key
    sys.stdout.flush()
    last_key = event.key
    if event.key == 'n':
        plt.close(fig)

folder = opt.eval_path
scenes = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
to_quit = False

for scene in scenes:
    t0_paths = sorted(glob.glob(os.path.join(folder, scene, 't0/*.png')))
    t1_paths = sorted(glob.glob(os.path.join(folder, scene, 't1/*.png')))
    t2_paths = sorted(glob.glob(os.path.join(folder, scene, 'groundtruth/*.png')))

    for i, (t0_path, t1_path, gt_path) in enumerate(zip(t0_paths, t1_paths, t2_paths)):
        t0 = cv2.cvtColor(cv2.imread(t0_path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        t1 = cv2.cvtColor(cv2.imread(t1_path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        gt = cv2.cvtColor(cv2.imread(gt_path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255

        # Resize the images for faster rendering
        t0 = cv2.resize(t0, dsize=(0, 0), fx=0.1, fy=0.1)
        t1 = cv2.resize(t1, dsize=(0, 0), fx=0.1, fy=0.1)
        gt = cv2.resize(gt, dsize=(0, 0), fx=0.1, fy=0.1)

        print('Displaying changes {}/{}...'.format(scene, i))
        print('Time 0: {}'.format(t0_path))
        print('Time 1: {}'.format(t1_path))
        print('Groundtruth: {}\n'.format(gt_path))

        # Overlays changes on the input images
        gray = np.tile(np.mean(t1, axis=2, keepdims=True), (1, 1, 3))
        gt2 = (gt * 0.7) + 0.3
        t1_wGt = gt2 * gray

        fig = plt.figure(figsize=(12, 3.5))
        fig.canvas.mpl_connect('key_press_event', on_press)
        plt.subplot(1, 3, 1)
        plt.imshow(t0)
        plt.title('Time 0')
        plt.subplot(1, 3, 2)
        plt.imshow(t1)
        plt.title('Time 1')
        plt.subplot(1, 3, 3)
        plt.imshow(t1_wGt)
        plt.title('Groundtruth changes (on T1)')
        plt.show()

        if last_key == 'q':
            to_quit = True
            break

    if to_quit:
        break
