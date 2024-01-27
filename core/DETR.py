import pytorch_lightning as pl
from transformers import DetrForObjectDetection
import torch


class Detr(pl.LightningModule):
    """
    Implementation of the DETR (DEtection TRansformer) model for object detection.
    """

    def __init__(self, lr, lr_backbone, weight_decay, train_dataloader, val_dataloader, id2label):
        """
        Initializes the DETR model.

        Args:
            lr (float): Learning rate for the optimizer.
            lr_backbone (float): Learning rate for the backbone.
            weight_decay (float): Weight decay for the optimizer.
            train_dataloader (torch.utils.data.DataLoader): Dataloader for training data.
            val_dataloader (torch.utils.data.DataLoader): Dataloader for validation data.
            id2label (dict): Mapping of class IDs to labels.
        """
        super().__init__()
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50",
                                                            num_labels=len(id2label),
                                                            ignore_mismatched_sizes=True)
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.trainDataloader = train_dataloader
        self.valDataloader = val_dataloader

    def forward(self, pixel_values, pixel_mask):
        """
        Forward pass of the DETR model.

        Args:
            pixel_values (torch.Tensor): Input image tensor.
            pixel_mask (torch.Tensor): Input mask tensor.

        Returns:
            outputs (torch.Tensor): Output tensor from the model.
        """
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)  

        return outputs

    def common_step(self, batch, batch_idx):
        """
        Common step for both training and validation steps.

        Args:
            batch (dict): Batch of data.
            batch_idx (int): Index of the batch.

        Returns:
            loss (torch.Tensor): Loss value.
            loss_dict (dict): Dictionary of individual loss values.
        """
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

        loss = outputs.loss
        loss_dict = outputs.loss_dict

        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        """
        Training step for the DETR model.

        Args:
            batch (dict): Batch of training data.
            batch_idx (int): Index of the batch.

        Returns:
            loss (torch.Tensor): Loss value.
        """
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("training_loss", loss)
        for k, v in loss_dict.items():
            self.log("train_" + k, v.item())

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the DETR model.

        Args:
            batch (dict): Batch of validation data.
            batch_idx (int): Index of the batch.

        Returns:
            loss (torch.Tensor): Loss value.
        """
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss)
        for k, v in loss_dict.items():
            self.log("validation_" + k, v.item())

        return loss

    def configure_optimizers(self):
        """
        Configures the optimizer for the DETR model.

        Returns:
            optimizer (torch.optim.Optimizer): Optimizer for the model.
        """
        param_dicts = [
            {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.lr_backbone,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)

        return optimizer

    def save_pretrained(self, path):
        """
        Saves the model to the given path.

        Args:
            path (str): Path to save the model to.

        Returns:
            None
        """
        self.model.save_pretrained(path)

    def train_dataloader(self):
        """
        Returns the dataloader for training data.

        Returns:
            trainDataloader (torch.utils.data.DataLoader): Dataloader for training data.
        """
        return self.trainDataloader

    def val_dataloader(self):
        """
        Returns the dataloader for validation data.

        Returns:
            valDataloader (torch.utils.data.DataLoader): Dataloader for validation data.
        """
        return self.valDataloader