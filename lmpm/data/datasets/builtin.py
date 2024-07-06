###########################################################################
# Created by: NTU
# Email: heshuting555@gmail.com
# Copyright (c) 2023
###########################################################################

import os
from .mevis import register_mevis_instances
from .ytvis import register_ytvis_instances

# ====    Predefined splits for mevis    ===========
_PREDEFINED_SPLITS_mevis = {
    "mevis_train": ("mevis/train",
                   "mevis/train/meta_expressions.json"),
    "mevis_val": ("mevis/valid_u",
                 "mevis/valid_u/meta_expressions.json"),
    "mevis_test": ("mevis/valid",
                  "mevis/valid/meta_expressions.json"),
}

_PREDEFINED_SPLITS_ytvis = {
     "ytvis_2021_train": ("ytvis_2021/train",
                    "ytvis_2021/train/meta_expressions.json"),
     "ytvis_2021_val": ("ytvis_2021/test",
                  "ytvis_2021/test/meta_expressions.json"),
     "ytvis_2021_test": ("ytvis_2021/valid",
                   "ytvis_2021/valid/meta_expressions.json"),
}

def register_all_mevis(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_mevis.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_mevis_instances(
            key,
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )

def register_all_ytvis_2021(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_ytvis.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )



if __name__.endswith(".builtin"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    register_all_mevis(_root)
    register_all_ytvis_2021(_root)



