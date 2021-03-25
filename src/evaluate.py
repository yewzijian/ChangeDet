"""Evaluate mIOU and F1 measures"""

import argparse
import os

from imageio import imread
import numpy as np

from utils import get_eval_list

THRESH = 127  # binary thresholding for change maps ranging from 0..255


parser = argparse.ArgumentParser()
parser.add_argument('--eval_path', type=str, default='../eval_data',
                    help='Folder containing images to be evaluated')
parser.add_argument('--result2d_path', type=str, default='../results2d/ours',
                    help='Output folder')
parser.add_argument('--num_change_classes', default=1, choices=[1, 2],
                    help='2 will differentiate appearance/disapperance.')
opt = parser.parse_args()


def convert_to_category(color_im, num_change_classes=1):
    """Convert color image representation to a single channel image containing
    the category code.
    """
    output = np.zeros(color_im.shape[0:2], dtype=np.uint8)
    if num_change_classes == 2:
        assert color_im.ndim == 3 and color_im.shape[2] == 2, 'num_change_classes can only be 2 if input has 3 channels'
        output[color_im[:, :, 2] > THRESH] = 1
        output[color_im[:, :, 0] > THRESH] = 2
    elif num_change_classes == 1:
        if color_im.ndim == 2:
            output = color_im > THRESH
        elif color_im.ndim == 3:
            output[np.any(color_im > THRESH, axis=-1)] = 1
        else:
            raise AssertionError('Change map must be of dimension 1 or 2')
    else:
        raise AssertionError('num_change_classes can only be 1 or 2.')
    return output


def compute_iou(groundtruth, pred):
    union = np.logical_or(groundtruth, pred)
    intersection = np.logical_and(groundtruth, pred)
    iou = np.sum(intersection, axis=(1, 2)) / np.sum(union, axis=(1, 2))
    return iou


def compute_tp_fp_fn(groundtruth, pred):

    true_pos = np.sum(np.logical_and(groundtruth, pred), axis=(1,2))
    false_pos = np.sum(np.logical_and(~groundtruth, pred), axis=(1,2))
    false_neg = np.sum(np.logical_and(groundtruth, ~pred), axis=(1,2))

    return true_pos, false_pos, false_neg


def compute_conf_matrix(gt, pred, num_classes):
    def _fast_hist(label_true, label_pred, n_class):
        """Fast computation of histogram
        Taken from KITTI vision benchmark code"""
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) +
            label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
        return hist

    hist = _fast_hist(gt.flatten(), pred.flatten(), num_classes)
    return hist


def compute_metrics(hist):
    """Adapted from KITTI vision benchmark code"""

    # For mIOU, we take the average of each class, including the background class
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)

    # For F1 score, we only consider the change classes
    tp = np.diag(hist)
    fp = hist.sum(axis=0) - np.diag(hist)
    fn = hist.sum(axis=1) - np.diag(hist)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    mean_f1 = np.mean(f1[1:])

    return {'Mean IoU': mean_iu, 'F1': mean_f1}


def evaluate(eval_path, eval_data, algo_result_path):

    scenes = eval_data.keys()

    overall_conf_matrix = np.zeros((opt.num_change_classes + 1, opt.num_change_classes + 1), np.int64)
    for scene in scenes:
        eval_scene_folder = os.path.join(eval_path, scene)
        eval_scene_data = eval_data[scene]
        pred_scene_folder = os.path.join(algo_result_path, scene)

        scene_conf_matrix = np.zeros((opt.num_change_classes + 1, opt.num_change_classes + 1), np.int64)

        for im0_fname, im1_fname, gt_fname in eval_scene_data:
            # Load raw images, groundtruth, and algo result
            im0 = imread(os.path.join(eval_scene_folder, 't0', im0_fname))
            im1 = imread(os.path.join(eval_scene_folder, 't1', im1_fname))
            gt = imread(os.path.join(eval_scene_folder, 'groundtruth', gt_fname))
            pred = imread(os.path.join(pred_scene_folder, gt_fname))

            # Convert color-coding to categories
            gt = convert_to_category(gt, num_change_classes=opt.num_change_classes)
            pred = convert_to_category(pred, num_change_classes=opt.num_change_classes)

            cur_conf_matrix = compute_conf_matrix(gt, pred, opt.num_change_classes + 1)
            cur_metrics = compute_metrics(cur_conf_matrix)
            print('{}/{}: {}'.format(scene, gt_fname, cur_metrics))

            scene_conf_matrix += cur_conf_matrix

        print('---------------------------------------------------------------')
        print('Overall {}: {}'.format(scene, compute_metrics(scene_conf_matrix)))
        print('---------------------------------------------------------------')
        overall_conf_matrix += scene_conf_matrix

    print('===============================================================')
    print('Overall: {}'.format(compute_metrics(overall_conf_matrix)))
    print('===============================================================')


if __name__ == '__main__':
    eval_data = get_eval_list(opt.eval_path)
    evaluate(opt.eval_path, eval_data, opt.result2d_path)