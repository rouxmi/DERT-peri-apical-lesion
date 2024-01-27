import numpy as np
import os
from PIL import Image, ImageDraw

def shows_example(train_dataset, img_folder):
    """
    Displays a random example image from the train dataset with bounding boxes and labels.

    Args:
        train_dataset (object): The train dataset object.
        img_folder (str): The path to the folder containing the train images.

    Returns:
        dict: A dictionary mapping category IDs to category names.
    """
    
    # based on https://github.com/woctezuma/finetune-detr/blob/master/finetune_detr.ipynb
    image_ids = train_dataset.coco.getImgIds()
    print(image_ids)
    # let's pick a random image
    image_id = image_ids[np.random.randint(0, len(image_ids))]
    print('Image n°{}'.format(image_id))
    image = train_dataset.coco.loadImgs(image_id)[0]
    image = Image.open(os.path.join(f'{img_folder}/train', image['file_name']))

    annotations = train_dataset.coco.imgToAnns[image_id]
    draw = ImageDraw.Draw(image)

    cats = train_dataset.coco.cats
    id2label = {k: v['name'] for k,v in cats.items()}

    for annotation in annotations:
        box = annotation['bbox']
        class_idx = annotation['category_id']
        x,y,x2,y2 = tuple(box)
        draw.rectangle((x,y,x2,y2), outline='red', width=3)
        draw.text((x, y), id2label[class_idx], fill='white')

    image.show()
    return id2label