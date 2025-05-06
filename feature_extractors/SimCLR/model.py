import torch
import torch.nn as nn
import pytorch_lightning as pl
from lightly.loss import NTXentLoss
from lightly.models.modules.heads import SimCLRProjectionHead

def uniformity_loss(z, t=2):
    z = torch.nn.functional.normalize(z, dim=1)
    pairwise_distances = torch.cdist(z, z, p=2)
    loss = -torch.log(torch.exp(-t * pairwise_distances ** 2).mean() + 1e-8)
    return loss


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet(nn.Module):
    def __init__(self, block, layers, in_channels=1, bottleneck_dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        current_channels = 64
        self.layer1, current_channels = self._make_layer(block, current_channels, 64, layers[0])
        self.layer2, current_channels = self._make_layer(block, current_channels, 128, layers[1], stride=2)
        self.layer3, current_channels = self._make_layer(block, current_channels, 256, layers[2], stride=2)
        self.layer4, current_channels = self._make_layer(block, current_channels, 256, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(current_channels, bottleneck_dim)

    def _make_layer(self, block, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = [block(in_channels, out_channels, stride, downsample)]
        in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(in_channels, out_channels))

        return nn.Sequential(*layers), in_channels

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)



class SimCLRModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()

        self.cfg = cfg

        encoder_cfg = cfg['encoder']
        head_cfg = cfg['projection_head']
        loss_cfg = cfg['loss']
        self.lambda_uniformity = loss_cfg['lambda_uniformity']

        if encoder_cfg['arch'] == "resnet":
            self.backbone = ResNet(
                block=BasicBlock,
                layers=encoder_cfg['resnet_layers'],
                in_channels=encoder_cfg['in_channels'],
                bottleneck_dim=encoder_cfg['bottleneck_dim']
            )
        else:
            raise ValueError(f"Unsupported encoder architecture: {encoder_cfg['arch']}")

        self.projection_head = SimCLRProjectionHead(
            encoder_cfg['bottleneck_dim'],
            head_cfg['hidden_dim'],
            head_cfg['output_dim']
        )

        self.criterion = NTXentLoss(temperature=loss_cfg['temperature'])

    def forward(self, x):
        h = self.backbone(x) 
        z = self.projection_head(h) 
        z = nn.functional.normalize(z, dim=1)
        return z

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        z0 = self(x0)
        z1 = self(x1)
        simclr_loss = self.criterion(z0, z1)
        
        z_all = torch.cat([z0, z1], dim=0)
        unif_loss = uniformity_loss(z_all)

        loss = simclr_loss + self.lambda_uniformity * unif_loss
        
        self.log("train_loss_ssl", loss, prog_bar=self.cfg['pl']['log_progress_bar'])
        self.log("train_loss_simclr", simclr_loss, prog_bar=self.cfg['pl']['log_progress_bar'])
        self.log("train_loss_uniformity", unif_loss, prog_bar=self.cfg['pl']['log_progress_bar'])
        return loss

    def validation_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        z0 = self(x0)
        z1 = self(x1)

        alignment = (z0 - z1).norm(dim=1).mean()

        z0_norm = nn.functional.normalize(z0, dim=1)
        pairwise_dist = torch.cdist(z0_norm, z0_norm, p=2)
        uniformity = torch.log(torch.exp(-2 * pairwise_dist ** 2).mean())

        self.log("val_alignment", alignment.item(), prog_bar=self.cfg['pl']['log_progress_bar'])
        self.log("val_uniformity", uniformity.item(), prog_bar=self.cfg['pl']['log_progress_bar'])

    def configure_optimizers(self):
        opt_cfg = self.cfg['optimizer']
        print(opt_cfg)
        optim = torch.optim.SGD(
            self.parameters(),
            lr=opt_cfg['lr'],
            momentum=opt_cfg['momentum'],
            weight_decay=opt_cfg['weight_decay']
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=opt_cfg['max_epochs'])
        return [optim], [scheduler]
