import contextlib
import io
import json
import logging
import numpy as np
from PIL import Image
import os
import pycocotools.mask as mask_util
from tqdm import tqdm
from fvcore.common.file_io import PathManager
from fvcore.common.timer import Timer

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
            vid_len = len(vid_frames)
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

        record["file_names"] = [os.path.join(image_root, 'JPEGImages', vid_dict['video'], str(frame) + '.jpg') for frame in vid_dict['frames'] ]
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
                mask_path = os.path.join(str(image_root), 'Annotations', video_name, str(frame) + '.png')
               
                bbox = [0, 0, 0, 0]
                obj["id"] = int(obj_id)
                obj["segmentation"] = mask_path
                obj["category_id"] = ytvos_category_dict[category]
                obj["bbox"] = bbox
                obj["bbox_mode"] = BoxMode.XYXY_ABS
                frame_objs.append(obj)
                video_objs.append(frame_objs)
        record["annotations"] = video_objs #list 
        record["sentence"] = exp
        record["exp_id"] = exp_id
        record["video_name"] = video_name
        dataset_dicts.append(record)

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
    meta = MetadataCatalog.get("ytvis_2019_train")

    json_file = "./datasets/ytvis/instances_train_sub.json"
    image_root = "./datasets/ytvis/train/JPEGImages"
    dicts = load_ytvis_json(json_file, image_root)
    logger.info("Done loading {} samples.".format(len(dicts)))

    dirname = "ytvis-data-vis"
    os.makedirs(dirname, exist_ok=True)

    def extract_frame_dic(dic, frame_idx):
        import copy
        frame_dic = copy.deepcopy(dic)
        annos = frame_dic.get("annotations", None)
        if annos is not None:
            frame_dic["annotations"] = annos[frame_idx]
        file_names = frame_dic.get("file_names", None)
        if file_names is not None:
            frame_dic["file_name"] = file_names[frame_idx]
        return frame_dic

    for d in dicts:
        for frame_idx in range(len(d["file_names"])):
            frame_dic = extract_frame_dic(d, frame_idx)
            img = np.array(Image.open(frame_dic["file_name"]))
            visualizer = Visualizer(img, metadata=meta)
            vis = visualizer.draw_dataset_dict(frame_dic)
            fpath = os.path.join(dirname, os.path.basename(frame_dic["file_name"]))
            vis.save(fpath)
            if frame_idx > 5:
                break
        break
