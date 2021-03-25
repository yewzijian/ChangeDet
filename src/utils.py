import os


def get_eval_list(eval_path):
    """Returns the list of images to evaluate

    Args:
        folder: Folder containing evaluation images

    """

    scenes = list(filter(lambda s: os.path.isdir(os.path.join(eval_path, s)),
                         os.listdir(eval_path)))

    eval_data = {}
    for scene in scenes:
        scene_folder = os.path.join(eval_path, scene)
        t0_folder = os.path.join(scene_folder, 't0')
        t1_folder = os.path.join(scene_folder, 't1')
        gt_folder = os.path.join(scene_folder, 'groundtruth')

        t0_paths = sorted(os.listdir(t0_folder))
        t1_paths = sorted(os.listdir(t1_folder))
        gt_paths = sorted(os.listdir(gt_folder))
        eval_data[scene] = list(zip(t0_paths, t1_paths, gt_paths))

    return eval_data
