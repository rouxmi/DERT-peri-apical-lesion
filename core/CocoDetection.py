import os
import torchvision

class CocoDetection(torchvision.datasets.CocoDetection):
    """
    Custom dataset class for COCO detection.

    Args:
        img_folder (str): Path to the folder containing the COCO images.
        feature_extractor: The feature extractor used for preprocessing the images and targets.
        train (bool, optional): Whether the dataset is for training or testing. Default is True.
    """

    def __init__(self, img_folder, feature_extractor, train=True):
        ann_file = os.path.join(img_folder, "train.json" if train else "test.json")
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.feature_extractor = feature_extractor

    def __getitem__(self, idx):
        """
        Retrieves the image and target at the given index.

        Args:
            idx (int): Index of the image and target to retrieve.

        Returns:
            tuple: A tuple containing the preprocessed image and target.
        """
        # read in PIL image and target in COCO format
        img, target = super(CocoDetection, self).__getitem__(idx)

        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        encoding = self.feature_extractor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze() # remove batch dimension
        target = encoding["labels"][0] # remove batch dimension

        return pixel_values, target