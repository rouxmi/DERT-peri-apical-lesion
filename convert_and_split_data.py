import datetime
import os
import configparser
import random
import json

import pandas as pd
from tqdm import tqdm
from PIL import Image

from core.convert import Dicom_to_png


if __name__=="__main__":
    # The goal of this script is to convert a collection of DICOM files to a collection of PNG files split into train and validation sets.
    # The script will also take a CSV file containing the labels for each image.
    # The script will also create a JSON file containing the labels for each image.

    config = configparser.ConfigParser()
    config.read("config.ini")
    image_folder = config["DEFAULT"]["img_folder"]
    contourage_folder = config["DEFAULT"]["contourage_folder"]
    image_size = {}
    uint = int(config["DEFAULT"]["uint"])
    width = int(config["DEFAULT"]["width"])
    height = int(config["DEFAULT"]["height"])
    
    # Create the train and validation folders
    if not os.path.exists(image_folder+'/train'):
        os.makedirs(image_folder+'/train')
    else:
        os.system(f'rm -rf {image_folder}/train/*')
    if not os.path.exists(image_folder+'/val'):
        os.makedirs(image_folder+'/val')
    else:
        os.system(f'rm -rf {image_folder}/val/*')
    
    # go through each image in the folder
    for image in tqdm(os.listdir(image_folder)):
        # skip the image if the image is a directory
        if os.path.isdir(f'{image_folder}/{image}'):
            continue
        # convert the image to PNG
        image_path = f'{image_folder}/{image}'

        # save the image to the correct folder
        if random.random() < float(config["DEFAULT"]["train_split"]):
            original_width, original_height = Dicom_to_png(image_path, f'{image_folder}/train/{image.split(".")[0]}.png', uint=uint, width=width, height=height)
        else:
            original_width, original_height = Dicom_to_png(image_path, f'{image_folder}/val/{image.split(".")[0]}.png', uint=uint, width=width, height=height)
        image_size[image.split(".")[0]] = (original_width, original_height)
        
        
    # generate the csv file with the bounding boxes
    # the csv file should have the following format:
    # image_name,x1,y1,x2,y2
    # where x1,y1 is the top left corner of the bounding box and x2,y2 is the bottom right corner of the bounding box
    
    df = pd.DataFrame(columns=['image_name', 'x1', 'y1', 'x2', 'y2'])
    for countour in tqdm(os.listdir(contourage_folder)):
        name = countour.split('.')[0].split('/')[-1]
        # check if the image is in the train or validation folder
        if name+".png" in os.listdir(f'{image_folder}/train'):
            name = f'{image_folder}'+ '/train/' + name + '.png'
        elif name+".png" in os.listdir(f'{image_folder}/val'):
            name = f'{image_folder}'+ '/val/' + name + '.png'
        else:
            continue
        x1 = 999999
        y1 = 999999
        x2 = 0
        y2 = 0
        with open(f'{contourage_folder}/{countour}', 'r') as f:
            lines = f.readlines()
            for line in lines:
                if '<Pt>' in line:
                    x, y = line.split('<Pt>')[1].split('</Pt>')[0].split(',')
                    x = int(float(x))
                    y = int(float(y))
                    if x < x1:
                        x1 = x
                    if x > x2:
                        x2 = x
                    if y < y1:
                        y1 = y
                    if y > y2:
                        y2 = y
                if '</Contour>' in line:
                    original_width, original_height = image_size[name.split('/')[-1].split('.')[0]]
                    x1 = (x1 - (original_width - width) / 2) / (2048 / width)
                    y1 = (y1 - (original_height - height) / 2) / (1024 / height)
                    x2 = (x2 - (original_width - width) / 2) / (2048 / width)
                    y2 = (y2 - (original_height - height) / 2) / (1024 / height)
                    df = df._append({'image_name': name, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}, ignore_index=True)
                    x1 = 999999
                    y1 = 999999
                    x2 = 0
                    y2 = 0
                    
    df.to_csv('annotate.txt', index=False, header=False)
        
    # times to create the JSON file for train and validation
    images_train_path = image_folder + '/train/'
    images_test_path = image_folder + '/val/'
    images_train = os.listdir(images_train_path)
    images_test = os.listdir(images_test_path)
    annotations = "annotate.txt"


    json_train = {}
    json_train['info'] = {
        "year": "2024",
        "version": "1.0",
        "description": "Periapical lesion radiographs",
        "contributor": "",
        "url": "",
        "date_created": str(datetime.datetime.now()),
    }

    json_train['licenses'] = []
    json_train['categories'] = [{"id": 0, "name": "lesion", "supercategory": None}]
    json_train['images'] = []
    json_train['annotations'] = []

    id_train = []
    for image in tqdm(images_train):
        id_train.append(image.split('.')[0])
        json_train['images'].append({
            "id": int(image.split('.')[0].split('/')[-1]),
            "file_name": image,
            "height": Image.open(images_train_path + image).size[1],
            "width": Image.open(images_train_path + image).size[0],
            "license": None,
            "coco_url": None
        })

    # open annotate.txt as a csv file
    id_annots = 0
    df = pd.read_csv(annotations, sep=',', header=None)
    for i in tqdm(range(len(df))):
        if df.iloc[i, 0].split('.')[0].split('/')[-1] in id_train:
            id_annots += 1
            json_train['annotations'].append({
                "id": int(id_annots),
                "image_id": int(df.iloc[i, 0].split('.')[0].split('/')[-1]),
                "category_id": 0,
                "bbox": [int(df.iloc[i, 1]), int(df.iloc[i, 2]), int(df.iloc[i, 3]), int(df.iloc[i, 4])], 
                "area": (float(df.iloc[i, 3]) - float(df.iloc[i, 1])) * (float(df.iloc[i, 4]) - float(df.iloc[i, 2]))
            }) 
            
    with open(images_train_path + 'train.json', 'w') as outfile:
        json.dump(json_train, outfile)
        
    # then the test folder
    json_test = {}
    json_test['info'] = {
        "year": "2024",
        "version": "1.0",
        "description": "Periapical lesion radiographs",
        "contributor": "",
        "url": "",
        "date_created": "22-01-2024",
    }

    json_test['licenses'] = []
    json_test['categories'] = [{"id": 0, "name": "lesion", "supercategory": None}]
    json_test['images'] = []
    json_test['annotations'] = []

    id_test = []
    for image in tqdm(images_test):
        id_test.append(image.split('.')[0])
        json_test['images'].append({
            "id": int(image.split('.')[0].split('/')[-1]),
            "file_name": image,
            "height": Image.open(images_test_path + image).size[1],
            "width": Image.open(images_test_path + image).size[0],
            "license": None,
            "coco_url": None
        })
        
    # open annotate.txt as a csv file
    id_annots = 0
    df = pd.read_csv(annotations, sep=',', header=None)
    for i in tqdm(range(len(df))):
        if df.iloc[i, 0].split('.')[0].split('/')[-1] in id_test:
            id_annots += 1
            json_test['annotations'].append({
                "id": int(id_annots),
                "image_id": int(df.iloc[i, 0].split('.')[0].split('/')[-1]),
                "category_id": 0,
                "bbox": [int(df.iloc[i, 1]), int(df.iloc[i, 2]), int(df.iloc[i, 3]), int(df.iloc[i, 4])],
                "area": (float(df.iloc[i, 3]) - float(df.iloc[i, 1])) * (float(df.iloc[i, 4]) - float(df.iloc[i, 2]))
            })
            
    with open(images_test_path + 'test.json', 'w') as outfile:
        json.dump(json_test, outfile)
        
    os.remove(annotations)
    print("Done !")