#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from argparse import ArgumentParser

import mmcv
from mmcv.utils import ProgressBar

from mmocr.apis import init_detector, model_inference
from mmocr.models import build_detector  # noqa: F401
from mmocr.utils import list_from_file, list_to_file
from mmocr.core import imshow_pred_boundary
import numpy as np
import cv2


def gen_target_path(target_root_path, src_name, suffix):
    """Gen target file path.

    Args:
        target_root_path (str): The target root path.
        src_name (str): The source file name.
        suffix (str): The suffix of target file.
    """
    assert isinstance(target_root_path, str)
    assert isinstance(src_name, str)
    assert isinstance(suffix, str)

    file_name = osp.split(src_name)[-1]
    name = osp.splitext(file_name)[0]
    return osp.join(target_root_path, name + suffix)


def save_results(result, out_dir, img_name, score_thr=0.3):
    """Save result of detected bounding boxes (quadrangle or polygon) to txt
    file.

    Args:
        result (dict): Text Detection result for one image.
        img_name (str): Image file name.
        out_dir (str): Dir of txt files to save detected results.
        score_thr (float, optional): Score threshold to filter bboxes.
    """
    assert 'boundary_result' in result
    assert score_thr > 0 and score_thr < 1

    txt_file = gen_target_path(out_dir, img_name, '.txt')
    valid_boundary_res = [
        res for res in result['boundary_result'] if res[-1] > score_thr
    ]
    lines = [
        ','.join([str(round(x)) for x in row]) for row in valid_boundary_res
    ]
    list_to_file(txt_file, lines)


def show_result_compare(img_path, out_img_name, result, annotations_root, compare_dir):
    """ Visual show image detection result compare with ground truth annotation.
    Args:
        img_path (str): Image path.
        out_img_name(str): the out compare image name.
        result(dict): result dict.
        annotations_root(str): Annotation file root.
        compare_dir: compare result output dir.

    Returns:
        None.
    """
    annotation_file = osp.join(annotations_root, 'gt_' + img_path.split('/')[-1].split('.')[0] + '.txt')
    if not osp.exists(annotation_file):
        print('{} not exists'.format(annotation_file))
        return
    # left show ground truth and right show detection result
    img_left = mmcv.imread(img_path)
    img_right = img_left.copy()
    img = np.concatenate((img_left, img_right), axis=1)
    # dontCare = transcription == "###"

    # draw ground truth
    boundaries = None
    labels = []
    if 'boundary_result' in result.keys():
        boundaries = result['boundary_result']
        labels = [0] * len(boundaries)
    with open(annotation_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        transcription = line.strip().split(',')[-1]
        if transcription == '###':
            # don't care
            continue
        boundaries_int = np.array(line.split(',')[:8]).astype(np.int32)
        cv2.polylines(img, [boundaries_int.reshape(-1, 1, 2)], True, color=(0, 255, 0), thickness=1)

    # draw detection result
    show_score = True
    if boundaries is not None:
        for boundary in boundaries:
            boundary[0] = boundary[0] + img_left.shape[1]
            boundary[2] = boundary[2] + img_left.shape[1]
            boundary[4] = boundary[4] + img_left.shape[1]
        imshow_pred_boundary(
            img,
            boundaries,
            labels,
            show=False,
            show_score=show_score)
    # save result
    out_img_path = osp.join(compare_dir, out_img_name)
    mmcv.imwrite(out_img_path, img)
    return


def main():
    parser = ArgumentParser()
    parser.add_argument('img_root', type=str, help='Image root path')
    parser.add_argument('img_list', type=str, help='Image path list file')
    parser.add_argument('config', type=str, help='Config file')
    parser.add_argument('checkpoint', type=str, help='Checkpoint file')
    parser.add_argument(
        '--score-thr', type=float, default=0.5, help='Bbox score threshold')
    parser.add_argument('--compare', action='store_true', help='Compare with ground truth')
    parser.add_argument('--annotations_root', type=str, default=None,
                        help='Annotations root path. When detection result compare with ground truth, it should be set.')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='./results',
        help='Dir to save '
        'visualize images '
        'and bbox')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference.')
    args = parser.parse_args()

    assert 0 < args.score_thr < 1
    assert args.compare and args.annotations_root is not None or not args.compare and args.annotations_root is None

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    if hasattr(model, 'module'):
        model = model.module

    # Start Inference
    out_vis_dir = osp.join(args.out_dir, 'out_vis_dir')
    mmcv.mkdir_or_exist(out_vis_dir)
    out_txt_dir = osp.join(args.out_dir, 'out_txt_dir')
    mmcv.mkdir_or_exist(out_txt_dir)
    if args.compare:
        compare_dir = osp.join(args.out_dir, 'compare_dir')
        mmcv.mkdir_or_exist(compare_dir)

    lines = list_from_file(args.img_list)
    progressbar = ProgressBar(task_num=len(lines))
    for idx, line in enumerate(lines):
        progressbar.update()
        img_path = osp.join(args.img_root, line.strip())
        if not osp.exists(img_path):
            raise FileNotFoundError(img_path)
        # Test a single image
        result = model_inference(model, img_path)
        img_name = osp.basename(img_path)
        img_name = "{:0>4}_".format(idx) + img_name
        # save result
        save_results(result, out_txt_dir, img_name, score_thr=args.score_thr)
        # show result
        out_file = osp.join(out_vis_dir, img_name)
        kwargs_dict = {
            'score_thr': args.score_thr,
            'show': False,
            'out_file': out_file
        }
        model.show_result(img_path, result, **kwargs_dict)
        # show result compare with ground truth
        if args.compare:
            show_result_compare(img_path, img_name, result, args.annotations_root,
                                compare_dir)

    print(f'\nInference done, and results saved in {args.out_dir}\n')


if __name__ == '__main__':
    main()
