from typing import List

import os
import shutil
import numpy as np

from lightly.active_learning.utils import BoundingBox
from lightly.active_learning.utils import ObjectDetectionOutput


def image_filename_to_label_filename(image_filename: str):
    """Converts the imagefilename to the label filename.
    
    """
    return f'{os.path.splitext(image_filename)[0]}.txt'


def load_model_outputs(al_agent, path: str, max_width: int = 720, max_height: int = 1280):
    """Loads the model outputs in the correct order and formats them for lightly.

    The model outputs need to be in the same order as the filenames in the query_tag
    of the active learning agent.

    The box coordinates must be normalized to [0, 1], hence max_width and max_height
    are passed as arguments.
    
    """
    outputs = []
    # iterate over all filenames in the query set and load the predictions
    # IMPORTANT: don't change the order of the predictions!
    for image_filename in al_agent.query_set:

        # get the filename of the inference from the image filename
        filename = image_filename_to_label_filename(image_filename)

        # load the inference from the file
        contents = np.genfromtxt(os.path.join(path, filename))
        
        if len(contents.shape) > 1:
            # in case there were predictions for this image, parse them
            boxes = []
            for x0, y0, x1, y1 in contents[:, 4:8]:
                box = BoundingBox(
                    max(0, min(x0, max_width)) / max_width,
                    max(0, min(y0, max_height)) / max_height,
                    min(max_width, x1) / max_width,
                    min(max_height, y1) / max_height,
                )
                boxes.append(box)

            # the objectness scores are in the last column
            scores = contents[:, -1]

            # in our case all labels are zero (everything is apples)
            labels = [0] * len(boxes)
        else:
            # case no predictions
            boxes = []
            scores = []
            labels = []

        # create an object detection output compatible with lightly
        model_output = ObjectDetectionOutput.from_scores(
            boxes,
            scores,
            labels,
        )
        
        outputs.append(model_output)

    return outputs


def annotate_images(filenames: List[str],
                    source_dir: str = 'data/raw',
                    target_dir: str = 'data/train',
                    image_dir: str = 'images',
                    label_dir: str = 'labels'):
    """Simulates the labeling process by copying images and labels from raw/ to train/.

    """

    # create target dirs if they don't exist yet
    os.makedirs(os.path.join(target_dir, image_dir), exist_ok=True)
    os.makedirs(os.path.join(target_dir, label_dir), exist_ok=True)
    
    for image_filename in filenames:
        
        label_filename = image_filename_to_label_filename(image_filename)
        
        # copy image
        shutil.copyfile(
            os.path.join(source_dir, image_dir, image_filename),
            os.path.join(target_dir, image_dir, image_filename)
        )

        # copy label
        shutil.copyfile(
            os.path.join(source_dir, label_dir, label_filename),
            os.path.join(target_dir, label_dir, label_filename)
        )