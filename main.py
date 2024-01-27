import os
import configparser

import torch
from transformers import DetrFeatureExtractor
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from PIL import Image

from core.CocoDetection import CocoDetection
from core.shows_examples import shows_example
from core.utility import *
from core.DETR import Detr


if __name__=="__main__":
    
    configparser = configparser.ConfigParser()
    configparser.read("config.ini")
    img_folder = configparser["DEFAULT"]["img_folder"]

    # Import a pretrained model that will be used for fine-tuning
    feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")

    train_dataset = CocoDetection(img_folder=f'{img_folder}/train', feature_extractor=feature_extractor)
    val_dataset = CocoDetection(img_folder=f'{img_folder}/val', feature_extractor=feature_extractor, train=False)

    print("Number of training examples:", len(train_dataset))
    print("Number of validation examples:", len(val_dataset))
    
    # Show a example to verify the data is correct
    id2label = shows_example(train_dataset, img_folder)
    
    def collate_fn(batch):
        pixel_values = [item[0] for item in batch]
        encoding = feature_extractor.pad(pixel_values, return_tensors="pt")
        labels = [item[1] for item in batch]
        batch = {}
        batch['pixel_values'] = encoding['pixel_values']
        batch['pixel_mask'] = encoding['pixel_mask']
        batch['labels'] = labels
        return batch
    
    # Create the dataloaders for the model
    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=4, shuffle=True)
    val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=2)
    batch = next(iter(train_dataloader))
    print(batch.keys())

    pixel_values, target = train_dataset[0]

    print(pixel_values.shape)

    print(target)

    model = Detr(lr=float(configparser["DEFAULT"]["lr"]), lr_backbone=float(configparser["DEFAULT"]["lr_backbone"]), weight_decay=float(configparser["DEFAULT"]["weight_decay"]),train_dataloader=train_dataloader, val_dataloader=val_dataloader, id2label=id2label) 

    outputs = model(pixel_values=batch['pixel_values'], pixel_mask=batch['pixel_mask'])

    print(outputs.logits.shape)

    """
    The stop condition for the training loop is the max_steps parameter.
    We can also use max_epochs instead to specify the number of epochs to train for.
    We can also use min_epochs to specify the minimum number of epochs to train for.
    We can also use max_time to specify the maximum amount of time to train for.
    We can also use limit_train_batches to specify the fraction of training data to use.
    We can also use limit_val_batches to specify the fraction of validation data to use.
    The stop condition can be a combination of the above parameters.
    
    For more derails, see https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-flags
    """
    trainer = Trainer(accelerator="auto", max_epochs=30, min_steps=1000)
    trainer.fit(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    it = iter(range(1500))

    #We can use the image_id in target to know which image it is
    pixel_values, target = val_dataset[next(it)]

    pixel_values = pixel_values.unsqueeze(0).to(device)
    print(pixel_values.shape)

    # forward pass to get class logits and bounding boxes
    outputs = model(pixel_values=pixel_values, pixel_mask=None)

    image_id = target['image_id'].item()
    image = val_dataset.coco.loadImgs(image_id)[0]
    image = Image.open(os.path.join(f'{img_folder}/val', image['file_name']))

    visualize_predictions(image, outputs, id2label, threshold=float(configparser["DEFAULT"]["threshold"]))

    # save the model to a detr folder
    model.save_pretrained("detr")
    
    