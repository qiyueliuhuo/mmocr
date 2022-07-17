# Copyright (c) OpenMMLab. All rights reserved.
"""
This script is used to convert the detection result json file to a txt file.
"""
from argparse import ArgumentParser
import argparse
import json


def main():
    parser = ArgumentParser()
    # eval-option rank_list
    parser.add_argument('eval_rank_list_file_path', type=str, help='Text detection test script with eval-option  '
                                                                   'rank_list parameter generate json file path')
    parser.add_argument('img_list', type=str, help='Image path list file')
    parser.add_argument('--ratio', type=float, default=None, optional=True, help='Convert detection result low ratio, '
                                                                   'when it\'s set, --hmean-thr is invalid')
    parser.add_argument(
        '--hmean-thr', type=float, default=0.5, help='Single image hmean threshold')

    args = parser.parse_args()
    assert args.eval_rank_list_file_path is not None and args.eval_rank_list_file_path.endswith('.json')
    assert args.img_list is not None and args.img_list.endswith('.txt')
    assert args.ratio or args.hmean_thr
    # read json file
    with open(args.eval_rank_list_file_path, 'r') as f:
        json_data = json.load(f)
    # convert json to txt
    ratio_num = None if args.ratio is None else len(json_data) * args.ratio # Convert ratio num to txt file.
    hmean_list = []
    with open(args.img_list, 'w') as f:
        for i, img_result in enumerate(json_data):  # In json file, Smaller hmean value in front.
            if ratio_num is not None and i <= ratio_num or img_result['hmean'] < args.hmean_thr:
                f.write(img_result['img_path'] + '\n')
            hmean_list.append(img_result['hmean'])
    print('The image min hmean:', min(hmean_list))
    print('The image max hmean:', max(hmean_list))
    print('The image avg hmean:', sum(hmean_list)/len(hmean_list))
    print('The image middle hmean num:', hmean_list[int(len(hmean_list)//2)])


if __name__ == '__main__':
    main()
