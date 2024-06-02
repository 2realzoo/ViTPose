# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp
import warnings
from collections import OrderedDict

import numpy as np
from mmcv import Config, deprecated_api_warning
from scipy.io import loadmat, savemat

from ...builder import DATASETS
from ..base import Kpt2dSviewRgbImgTopDownDataset


@DATASETS.register_module()
class TopDownPictogramDataset(Kpt2dSviewRgbImgTopDownDataset):
    """Pictogram Dataset for top-down pose estimation.

    "2D Human Pose Estimation: New Benchmark and State of the Art Analysis"
    ,CVPR'2014. More details can be found in the `paper
    <http://human-pose.mpi-inf.mpg.de/contents/andriluka14cvpr.pdf>`__ .

    The dataset loads raw features and apply specified transforms
    to return a dict containing the image tensors and other information.

    MPII keypoint indexes::

        0: 'right_ankle'
        1: 'right_knee',
        2: 'right_hip',
        3: 'left_hip',
        4: 'left_knee',
        5: 'left_ankle',
        6: 'pelvis',
        7: 'thorax',
        8: 'upper_neck',
        9: 'head_top',
        10: 'right_wrist',
        11: 'right_elbow',
        12: 'right_shoulder',
        13: 'left_shoulder',
        14: 'left_elbow',
        15: 'left_wrist'

    Args:
        ann_file (str): Path to the annotation file.
        img_prefix (str): Path to a directory where images are held.
            Default: None.
        data_cfg (dict): config
        pipeline (list[dict | callable]): A sequence of data transforms.
        dataset_info (DatasetInfo): A class containing all dataset info.
        test_mode (bool): Store True when building test or
            validation dataset. Default: False.
    """

    def __init__(self,
                 ann_file,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 dataset_info=None,
                 test_mode=False):

        if dataset_info is None:
            warnings.warn(
                'dataset_info is missing. '
                'Check https://github.com/open-mmlab/mmpose/pull/663 '
                'for details.', DeprecationWarning)
            cfg = Config.fromfile('configs/_base_/datasets/mpii.py')
            dataset_info = cfg._cfg_dict['dataset_info']

        super().__init__(
            ann_file,
            img_prefix,
            data_cfg,
            pipeline,
            dataset_info=dataset_info,
            coco_style=False,
            test_mode=test_mode)

        self.db = self._get_db()
        self.image_set = set(x['image_file'] for x in self.db)
        self.num_images = len(self.image_set)

        print(f'=> num_images: {self.num_images}')
        print(f'=> load {len(self.db)} samples')

    def _get_db(self):
        # create train/val split
        with open(self.ann_file) as anno_file:
            anno = json.load(anno_file)

        gt_db = []
        bbox_id = 0
        for a in anno:
            image_name = a['image']

            center = np.array(a['center'], dtype=np.float32)
            scale = np.array([a['scale'], a['scale']], dtype=np.float32)

            # Adjust center/scale slightly to avoid cropping limbs
            if center[0] != -1:
                center[1] = center[1] + 15 * scale[1]
                # padding to include proper amount of context
                scale = scale * 1.25

            # MPII uses matlab format, index is 1-based,
            # we should first convert to 0-based index
            center = center - 1

            joints_3d = np.zeros((self.ann_info['num_joints'], 3),
                                 dtype=np.float32)
            joints_3d_visible = np.zeros((self.ann_info['num_joints'], 3),
                                         dtype=np.float32)
            if not self.test_mode:
                joints = np.array(a['joints'])
                joint_vis = np.array(a['joint_vis'])
                assert len(joints) == self.ann_info['num_joints'], \
                    f'joint num diff: {len(joints)}' + \
                    f' vs {self.ann_info["num_joints"]}'

                joints_3d[:, 0:2] = joints[:, 0:2] - 1
                joints_3d_visible[:, :2] = joint_vis[:, None]
            image_file = osp.join(self.img_prefix, image_name)
            gt_db.append({
                'image_file': image_file,
                'bbox_id': bbox_id,
                'center': center,
                'scale': scale,
                'rotation': 0,
                'joints_3d': joints_3d,
                'joints_3d_visible': joints_3d_visible,
                'dataset': self.dataset_name,
                'bbox_score': 1
            })
            bbox_id = bbox_id + 1
        gt_db = sorted(gt_db, key=lambda x: x['bbox_id'])

        return gt_db

    @deprecated_api_warning(name_dict=dict(outputs='results'))
    def evaluate(self, results, res_folder=None, metric='PCK', **kwargs):
        """Evaluate PCKh for MPII dataset. Adapted from
        https://github.com/leoxiaobin/deep-high-resolution-net.pytorch
        Copyright (c) Microsoft, under the MIT License.

        Note:
            - batch_size: N
            - num_keypoints: K
            - heatmap height: H
            - heatmap width: W

        Args:
            results (list[dict]): Testing results containing the following
                items:

                - preds (np.ndarray[N,K,3]): The first two dimensions are \
                    coordinates, score is the third dimension of the array.
                - boxes (np.ndarray[N,6]): [center[0], center[1], scale[0], \
                    scale[1],area, score]
                - image_paths (list[str]): For example, ['/val2017/000000\
                    397133.jpg']
                - heatmap (np.ndarray[N, K, H, W]): model output heatmap.
            res_folder (str, optional): The folder to save the testing
                results. Default: None.
            metric (str | list[str]): Metrics to be performed.
                Defaults: 'PCKh'.

        Returns:
            dict: PCKh for each joint
        """

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['PCK']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported', metrics)

        kpts = []
        print('result : ',result)
        for result in results:
            preds = result['preds']
            bbox_ids = result['bbox_ids']
            batch_size = len(bbox_ids)
            for i in range(batch_size):
                kpts.append({'keypoints': preds[i], 'bbox_id': bbox_ids[i]})
        kpts = self._sort_and_unique_bboxes(kpts)

        preds = np.stack([kpt['keypoints'] for kpt in kpts])
        print('preds : ', preds)
        # convert 0-based index to 1-based index,
        # and get the first two dimensions.
        preds = preds[..., :2] + 1.0
        print('changed-preds : ', preds)
        if res_folder:
            pred_file = osp.join(res_folder, 'pred.mat')
            savemat(pred_file, mdict={'preds': preds})

        SC_BIAS = 0.6
        threshold = 0.5

        gt_file = osp.join(osp.dirname(self.ann_file), 'test_annotations.json')
        with open(gt_file, 'r') as file:
            data = json.load(file)
        dataset_joints = np.array(data[0]['joints'])
        jnt_visible = np.array(data[0]['joint_vis'])
        center = np.array(data[0]['center'])
        scale = np.array(data[0]['scale'])

        pos_pred_src = np.transpose(preds, [1, 2, 0])
        print('pos_pred_src: ', pos_pred_src)
        
        head = [9]
        lsho = [13]
        lelb = [14]
        lwri = [15]
        lhip = [3]
        lkne = [4]
        lank = [5]

        rsho = [12]
        relb = [11]
        rwri = [10]
        rkne = [1]
        rank = [0]
        rhip = [2]

        uv_error = pos_pred_src - dataset_joints
        uv_err = np.linalg.norm(uv_error, axis=1)
        # headsizes = headboxes_src[1, :, :] - headboxes_src[0, :, :]
        # headsizes = np.linalg.norm(headsizes, axis=0)
        # headsizes *= SC_BIAS
        # scale = headsizes * np.ones((len(uv_err), 1), dtype=np.float32)
        scaled_uv_err = uv_err / scale
        scaled_uv_err = scaled_uv_err * jnt_visible
        jnt_count = np.sum(jnt_visible, axis=1)
        less_than_threshold = (scaled_uv_err <= threshold) * jnt_visible
        PCK = 100. * np.sum(less_than_threshold, axis=1) / jnt_count

        # save
        rng = np.arange(0, 0.5 + 0.01, 0.01)
        pckAll = np.zeros((len(rng), 16), dtype=np.float32)

        for r, threshold in enumerate(rng):
            less_than_threshold = (scaled_uv_err <= threshold) * jnt_visible
            pckAll[r, :] = 100. * np.sum(
                less_than_threshold, axis=1) / jnt_count

        PCK = np.ma.array(PCK, mask=False)
        PCK.mask[6:8] = True

        jnt_count = np.ma.array(jnt_count, mask=False)
        jnt_count.mask[6:8] = True
        jnt_ratio = jnt_count / np.sum(jnt_count).astype(np.float64)

        name_value = [('Head', PCK[head]),
                      ('Shoulder', 0.5 * (PCK[lsho] + PCK[rsho])),
                      ('Elbow', 0.5 * (PCK[lelb] + PCK[relb])),
                      ('Wrist', 0.5 * (PCK[lwri] + PCK[rwri])),
                      ('Hip', 0.5 * (PCK[lhip] + PCK[rhip])),
                      ('Knee', 0.5 * (PCK[lkne] + PCK[rkne])),
                      ('Ankle', 0.5 * (PCK[lank] + PCK[rank])),
                      ('PCK', np.sum(PCK * jnt_ratio)),
                      ('PCK@0.1', np.sum(pckAll[10, :] * jnt_ratio))]
        name_value = OrderedDict(name_value)

        return name_value

    def _sort_and_unique_bboxes(self, kpts, key='bbox_id'):
        """sort kpts and remove the repeated ones."""
        kpts = sorted(kpts, key=lambda x: x[key])
        num = len(kpts)
        for i in range(num - 1, 0, -1):
            if kpts[i][key] == kpts[i - 1][key]:
                del kpts[i]

        return kpts
