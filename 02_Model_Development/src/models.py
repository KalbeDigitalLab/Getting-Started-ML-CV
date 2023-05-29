from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics as tm
from lightning import LightningModule
import timm


class BasicBlock(nn.Module):
    """ResNet Basic Block.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    stride : int, optional
        Convolution stride size, by default 1
    identity_downsample : Optional[torch.nn.Module], optional
        Downsampling layer, by default None
    """

    def __init__(self,
                in_channels: int,
                out_channels: int,
                stride: int = 1,
                identity_downsample: Optional[torch.nn.Module] = None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size = 3,
                              stride = stride,
                              padding = 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels,
                              out_channels,
                              kernel_size = 3,
                              stride = 1,
                              padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.identity_downsample = identity_downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply forward computation."""
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        # Apply an operation to the identity output.
        # Useful to reduce the layer size and match from conv2 output
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x += identity
        x = self.relu(x)
        return x


class ResNet18(nn.Module):
    """Construct ResNet-18 Model.

    Parameters
    ----------
    input_channels : int
        Number of input channels
    num_classes : int
        Number of class outputs
    """

    def __init__(self, input_channels, num_classes):

        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(input_channels,
                               64, kernel_size = 7,
                              stride = 2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size = 3,
                                   stride = 2,
                                   padding = 1)

        self.layer1 = self._make_layer(64, 64, stride = 1)
        self.layer2 = self._make_layer(64, 128, stride = 2)
        self.layer3 = self._make_layer(128, 256, stride = 2)
        self.layer4 = self._make_layer(256, 512, stride = 2)

        # Last layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def identity_downsample(self, in_channels: int, out_channels: int) -> nn.Module:
        """Downsampling block to reduce the feature sizes."""
        return nn.Sequential(
             nn.Conv2d(in_channels,
                       out_channels,
                       kernel_size = 3,
                       stride = 2,
                       padding = 1),
            nn.BatchNorm2d(out_channels)
        )

    def _make_layer(self, in_channels: int, out_channels: int, stride: int) -> nn.Module:
        """Create sequential basic block."""
        identity_downsample = None

        # Add downsampling function
        if stride != 1:
            identity_downsample = self.identity_downsample(in_channels, out_channels)

        return nn.Sequential(
                    BasicBlock(in_channels, out_channels, identity_downsample=identity_downsample, stride=stride),
                    BasicBlock(out_channels, out_channels)
                    )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


class TimmModel(nn.Module):
    """Construct Timm Based Model.

    Parameters
    ----------
    input_channels : int
        Number of input channels
    num_classes : int
        Number of class outputs
    model_name: str
        Model type that is going to use, by default efficientnetv2_rw_s
    pretrained: bool
        Use pretrained model, by default True
    """

    def __init__(self,
                 input_channels: int,
                 num_classes: int,
                 model_name: str = "efficientnetv2_rw_s",
                 pretrained : bool = True
    ):
        super(TimmModel, self).__init__()
        self.model = timm.create_model(
            model_name, pretrained=pretrained,
            num_classes = num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


class ClassificationLightningModule(LightningModule):
    """Model training pipeline for Food101 classification.

    Parameters
    ----------
    net : torch.nn.Module
        The model module or configuration
    num_classes : int, optional
        Number of output classes, by default 10
    lr : float, optional
        Optimizer learning rate, by default 0.00001
    weight_decay : float, optional
        Optimizer weight decay, by default 0.0005
    """

    def __init__(
        self,
        net: torch.nn.Module,
        num_classes: int = 10,
        lr: float = 0.00001,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = tm.Accuracy(task='multiclass', num_classes=num_classes)

        self.val_metrics = tm.MetricCollection({
            'acc': tm.Accuracy(task='multiclass', num_classes=num_classes),
            'prec': tm.Precision(task='multiclass', num_classes=num_classes),
            'rec': tm.Recall(task='multiclass', num_classes=num_classes),
            'auroc': tm.AUROC(task='multiclass', num_classes=num_classes),
            'f1': tm.F1Score(task='multiclass', num_classes=num_classes),
        })  # type: ignore

        self.test_metrics = self.val_metrics.clone()

    def reset_metrics(self):
        self.train_acc.reset()
        self.val_metrics.reset()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.reset_metrics()

    def model_step(self, batch: Any):
        images, targets = batch
        targets = targets.squeeze().long() # convert to 1D
        logits = self.forward(images)
        loss = self.criterion(logits, targets)
        preds = F.softmax(logits, dim=1)
        return loss, preds, targets

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        acc = self.train_acc(preds, targets)
        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_start(self):
        self.reset_metrics()

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        metrics = self.val_metrics(preds, targets)
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/acc', metrics['acc'], on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/prec', metrics['prec'], on_step=False, on_epoch=True, prog_bar=False)
        self.log('val/rec', metrics['rec'], on_step=False, on_epoch=True, prog_bar=False)
        self.log('val/auroc', metrics['auroc'], on_step=False, on_epoch=True, prog_bar=False)
        self.log('val/f1', metrics['f1'], on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        metrics = self.test_metrics(preds, targets)
        self.log('test/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test/acc', metrics['acc'], on_step=False, on_epoch=True, prog_bar=True)
        self.log('test/prec', metrics['prec'], on_step=False, on_epoch=True, prog_bar=False)
        self.log('test/rec', metrics['rec'], on_step=False, on_epoch=True, prog_bar=False)
        self.log('test/auroc', metrics['auroc'], on_step=False, on_epoch=True, prog_bar=False)
        self.log('test/f1', metrics['f1'], on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        """Configure the optimizer and scheduler to use.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr=self.hparams.lr,
        )

        return optimizer
