from typing import List

import os
import shutil

import helpers


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
        
        label_filename = helpers.image_filename_to_label_filename(image_filename)
        
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
