# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/sukjunhwang/IFC

import contextlib
import io
import json
import logging
import numpy as np
from PIL import Image
import os
import pycocotools.mask as mask_util
from fvcore.common.file_io import PathManager
from fvcore.common.timer import Timer
from tqdm import tqdm

from detectron2.structures import Boxes, BoxMode, PolygonMasks
from detectron2.data import DatasetCatalog, MetadataCatalog

"""
This file contains functions to parse YTVIS dataset of
COCO-format annotations into dicts in "Detectron2 format".
"""

logger = logging.getLogger(__name__)

__all__ = ["load_ytvis_json", "register_ytvis_instances"]


ytvos_category_dict = {
    'airplane': 0, 'ape': 1, 'bear': 2, 'bike': 3, 'bird': 4, 'boat': 5, 'bucket': 6, 'bus': 7, 'camel': 8, 'cat': 9, 
    'cow': 10, 'crocodile': 11, 'deer': 12, 'dog': 13, 'dolphin': 14, 'duck': 15, 'eagle': 16, 'earless_seal': 17, 
    'elephant': 18, 'fish': 19, 'fox': 20, 'frisbee': 21, 'frog': 22, 'giant_panda': 23, 'giraffe': 24, 'hand': 25, 
    'hat': 26, 'hedgehog': 27, 'horse': 28, 'knife': 29, 'leopard': 30, 'lion': 31, 'lizard': 32, 'monkey': 33, 
    'motorbike': 34, 'mouse': 35, 'others': 36, 'owl': 37, 'paddle': 38, 'parachute': 39, 'parrot': 40, 'penguin': 41, 
    'person': 42, 'plant': 43, 'rabbit': 44, 'raccoon': 45, 'sedan': 46, 'shark': 47, 'sheep': 48, 'sign': 49, 
    'skateboard': 50, 'snail': 51, 'snake': 52, 'snowboard': 53, 'squirrel': 54, 'surfboard': 55, 'tennis_racket': 56, 
    'tiger': 57, 'toilet': 58, 'train': 59, 'truck': 60, 'turtle': 61, 'umbrella': 62, 'whale': 63, 'zebra': 64
}

def load_ytvis_json(image_root, json_file):
    num_instances_without_valid_segmentation = 0
    num_instances_valid_segmentation = 0

    ann_file = json_file
    # read object information

    with open(os.path.join(str(image_root), 'meta.json'), 'r') as f:
        subset_metas_by_video = json.load(f)['videos']
    
    # read expression data: meta_expressions.json
    with open(str(ann_file), 'r') as f:
        subset_expressions_by_video = json.load(f)['videos']
    videos = list(subset_expressions_by_video.keys())

    metas = []
    if image_root.split('/')[-1] == 'train':
        for vid in videos:
            vid_meta = subset_metas_by_video[vid]
            vid_data = subset_expressions_by_video[vid]
            vid_frames = sorted(vid_data['frames'])
            vid_len = int(subset_expressions_by_video[vid]['frames'][-1]) + 1 
            for exp_id, exp_dict in vid_data['expressions'].items():
                meta = {}
                meta['video'] = vid
                meta['exp'] = exp_dict['exp']
                meta['obj_id'] = int(exp_dict['obj_id'])
                meta['frames'] = vid_frames
                # get object category
                obj_id = exp_dict['obj_id']
                meta['category'] = vid_meta['objects'][obj_id]['category']
                meta['length'] = vid_len
                metas.append(meta)
    else: 
        NotImplementedError
    dataset_dicts = []

    
    for vid_dict in tqdm(metas):
        record = {}
        files_name = [file_name_i for file_name_i in os.listdir(os.path.join(image_root, 'JPEGImages', vid_dict['video']))] 
        files_name.sort()
        record["file_names"] = [os.path.join(image_root, 'JPEGImages', vid_dict['video'], {frame} + '.jpg') for frame in vid_dict['frames'] ]
        record["length"] = vid_dict["length"]
    

        video_name, exp, obj_id, category, frames = \
                    vid_dict['video'], vid_dict['exp'], vid_dict['obj_id'], vid_dict['category'], vid_dict['frames']
        # clean up the caption
        exp = " ".join(exp.lower().split())
        
        video_objs = []
        if image_root.split('/')[-1] == 'train':
            for frame in vid_dict['frames']:
                frame_objs = []
                
                obj = {}
                mask_path = os.path.join(str(self.img_folder), 'Annotations', video, frame_name + '.png')
                mask = Image.open(mask_path).convert('P')

                mask = np.array(mask)
                mask = (mask==obj_id).astype(np.float32) # 0,1 binary
                
                if not (mask > 0).any():
                    num_instances_without_valid_segmentation += 1
                    continue
                num_instances_valid_segmentation += 1
                bbox = [0, 0, 0, 0]
                obj["id"] = obj_id
                obj["segmentation"] = mask
                obj["category_id"] = category
                obj["bbox"] = bbox
                obj["bbox_mode"] = BoxMode.XYXY_ABS
                frame_objs.append(obj)
                video_objs.append(frame_objs)
        record["annotations"] = video_objs
        record["sentence"] = exp
        record["exp_id"] = exp_id
        record["video_name"] = video_name
        dataset_dicts.append(record)

    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            "Total {} instance and Filtered out {} instances without valid segmentation. ".format(
                num_instances_valid_segmentation, num_instances_without_valid_segmentation
            )
        )
    return dataset_dicts


def register_ytvis_instances(name, json_file, image_root):
    """
    Register a dataset in YTVIS's json annotation format for
    instance tracking.

    Args:
        name (str): the name that identifies a dataset, e.g. "ytvis_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: load_ytvis_json(image_root, json_file))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="ytvis"
    )


if __name__ == "__main__":
    """
    Test the YTVIS json dataset loader.
    """
    from detectron2.utils.logger import setup_logger
    from detectron2.utils.visualizer import Visualizer
    import detectron2.data.datasets  # noqa # add pre-defined metadata
    import sys
    from PIL import Image

    logger = setup_logger(name=__name__)
    #assert sys.argv[3] in DatasetCatalog.list()
    meta = MetadataCatalog.get("ytvis_2019_train")

    json_file = "./datasets/ytvis/instances_train_sub.json"
    image_root = "./datasets/ytvis/train/JPEGImages"
    dicts = load_ytvis_json(json_file, image_root, dataset_name="ytvis_2019_train")
    logger.info("Done loading {} samples.".format(len(dicts)))

    dirname = "ytvis-data-vis"
    os.makedirs(dirname, exist_ok=True)

    def extract_frame_dic(dic, frame_idx):
        import copy
        frame_dic = copy.deepcopy(dic)
        annos = frame_dic.get("annotations", None)
        if annos:
            frame_dic["annotations"] = annos[frame_idx]

        return frame_dic

    for d in dicts:
        vid_name = d["file_names"][0].split('/')[-2]
        os.makedirs(os.path.join(dirname, vid_name), exist_ok=True)
        for idx, file_name in enumerate(d["file_names"]):
            img = np.array(Image.open(file_name))
            visualizer = Visualizer(img, metadata=meta)
            vis = visualizer.draw_dataset_dict(extract_frame_dic(d, idx))
            fpath = os.path.join(dirname, vid_name, file_name.split('/')[-1])
            vis.save(fpath)