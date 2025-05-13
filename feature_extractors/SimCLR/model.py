import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from lightly.loss import NTXentLoss
from lightly.models.modules.heads import SimCLRProjectionHead
from torchmetrics.classification import BinaryAUROC, BinaryF1Score, BinaryPrecision, BinaryRecall
from collections import defaultdict

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



class TeacherModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        
        self.cfg = cfg

        self.image_encoder = nn.Sequential(
            nn.Conv2d(self.cfg['in_channels'], 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 + self.cfg['covariate_dim'], self.cfg['hidden_dim']),
            nn.ReLU(),
            nn.Linear(self.cfg['hidden_dim'], 2)
        )

        self.loss_fn = nn.CrossEntropyLoss()
        
        self.auroc = BinaryAUROC()
        self.f1 = BinaryF1Score()
        self.precision = BinaryPrecision()
        self.recall = BinaryRecall()

    def forward(self, image, covariates):
        feat = self.image_encoder(image)
        xz = torch.cat([feat, covariates.squeeze()], dim=1)
        return self.classifier(xz)

    def training_step(self, batch, batch_idx):
        image, x, z, y = batch['image'], batch['x'], batch['z'], batch['d'].squeeze()
        covars = torch.stack([x, z], dim=1).float()
        logits = self(image, covars)
        loss = self.loss_fn(logits, y.long())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=self.cfg['pl']['log_progress_bar'])
        return loss

    def validation_step(self, batch, batch_idx):
        image, x, z, y = batch['image'], batch['x'], batch['z'], batch['d'].squeeze()
        covars = torch.stack([x, z], dim=1).float()
        logits = self(image, covars)
        loss = self.loss_fn(logits, y.long())
        
        preds = torch.softmax(logits, dim=1)[:, 1]
        class_preds = (preds > 0.5).long()

        acc = (class_preds == y).float().mean()
        self.auroc.update(preds, y)
        self.f1.update(class_preds, y)
        self.precision.update(class_preds, y)
        self.recall.update(class_preds, y)
        
        self.log_dict({"val_loss": loss, "val_acc": acc}, on_epoch=True, prog_bar=self.cfg['pl']['log_progress_bar'])
        
    def on_validation_epoch_end(self):
        self.log_dict({
            "val_auroc": self.auroc.compute(),
            "val_f1": self.f1.compute(),
            "val_precision": self.precision.compute(),
            "val_recall": self.recall.compute(),
        }, on_epoch=True, prog_bar=self.cfg['pl'].get('log_progress_bar', True))

        self.auroc.reset()
        self.f1.reset()
        self.precision.reset()
        self.recall.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.cfg['lr'], weight_decay=self.cfg['weight_decay'])
    
class StudentModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        
        self.cfg = cfg
        
        self.encoder = SimCLRModel.load_from_checkpoint(cfg['encoder_ckpt'])
        encoder_dim = self.encoder.cfg['encoder']['bottleneck_dim']
        self.encoder = self.encoder.backbone
    
        
        self.classifier = nn.Sequential(
            nn.Linear(encoder_dim+cfg['covariate_dim'], cfg['hidden_dim']),
            nn.ReLU(),
            nn.Linear(cfg['hidden_dim'], 2)
        )
    
    def forward(self, image, covariates):
        z = F.normalize(self.encoder(image))
        xz = torch.cat([z, covariates.squeeze()], dim=1)
        return self.classifier(xz), z 
    
class StudentTrainer(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()

        self.cfg = cfg
        self.student = StudentModel(cfg)

        self.teacher = TeacherModel.load_from_checkpoint(cfg['teacher_ckpt'])
        self.teacher.freeze()

        self.loss_supervised = nn.CrossEntropyLoss()
        
        self.auroc = BinaryAUROC()
        self.f1 = BinaryF1Score()
        self.precision = BinaryPrecision()
        self.recall = BinaryRecall()
    
    def js_divergence(self, student_logits, teacher_logits):
        p = F.softmax(student_logits, dim=1)
        q = F.softmax(teacher_logits, dim=1)
        m = 0.5 * (p + q)
        return 0.5 * (F.kl_div(p.log(), m, reduction='batchmean') +
                    F.kl_div(q.log(), m, reduction='batchmean'))

    def training_step(self, batch, batch_idx):
        image, x, z, y = batch['image'], batch['x'], batch['z'], batch['d'].squeeze()
        covars = torch.stack([x, z], dim=1).float()

        logits_student, emb = self.student(image, covars)
        logits_teacher = self.teacher(image, covars).detach()
        
        assert not torch.isnan(emb).any(), "NaNs in embeddings!"

        loss_sup = self.loss_supervised(logits_student, y.long())
        loss_align = self.js_divergence(logits_student, logits_teacher)

        loss = self.cfg['lambda_sup'] * loss_sup + self.cfg['lambda_align'] * loss_align

        self.log_dict({
            "train_loss": loss,
            "loss_sup": loss_sup,
            "loss_align": loss_align
        }, on_epoch=True, prog_bar=self.cfg['pl']['log_progress_bar'])
        return loss

    def validation_step(self, batch, batch_idx):
        image, x, z, y = batch['image'], batch['x'], batch['z'], batch['d'].squeeze()
        covars = torch.stack([x, z], dim=1).float()

        logits_student, emb = self.student(image, covars)
        logits_teacher = self.teacher(image, covars).detach()

        loss_sup = self.loss_supervised(logits_student, y.long())
        loss_align = self.js_divergence(logits_student, logits_teacher)
        
        assert not torch.isnan(emb).any(), "NaNs in embeddings!"

        loss = self.cfg['lambda_sup'] * loss_sup + self.cfg['lambda_align'] * loss_align

        preds = torch.softmax(logits_student, dim=1)[:, 1]
        class_preds = (preds > 0.5).long()
        acc = (class_preds == y).float().mean()
        
        self.auroc.update(preds, y)
        self.f1.update(class_preds, y)
        self.precision.update(class_preds, y)
        self.recall.update(class_preds, y)
        

        self.log_dict({
            "val_loss": loss,
            "loss_sup": loss_sup,
            "loss_align": loss_align,
            "val_acc": acc
        }, on_epoch=True, prog_bar=self.cfg['pl']['log_progress_bar'])
        return loss
    
    def on_validation_epoch_end(self):
        self.log_dict({
            "val_auroc": self.auroc.compute(),
            "val_f1": self.f1.compute(),
            "val_precision": self.precision.compute(),
            "val_recall": self.recall.compute(),
        }, on_epoch=True)

        self.auroc.reset()
        self.f1.reset()
        self.precision.reset()
        self.recall.reset()
    
    def extract_embeddings(self, dataloader):
        self.eval()
        all_data = defaultdict(list)
        device = self.device
        
        with torch.no_grad():
            for batch in dataloader:
                image = batch['image'].to(device)
                x = batch['x'].to(device)
                z = batch['z'].to(device)
                covars = torch.stack([x, z], dim=1).float()

                logits, embeddings = self.student(image, covars)

                assert not torch.isnan(embeddings).any(), "NaNs in embeddings!"

                for key, val in batch.items():
                    all_data[key].append(val.cpu())
                all_data['embedding'].append(embeddings.cpu())
        
        final_data = {k: torch.cat(v, dim=0) for k, v in all_data.items()}
        return final_data
        

    def configure_optimizers(self):
        encoder_params = list(self.student.encoder.parameters())
        classifier_params = list(self.student.classifier.parameters())

        optimizer = torch.optim.Adam([
            {'params': encoder_params, 'lr': self.cfg['lr'] * 0.1},
            {'params': classifier_params, 'lr': self.cfg['lr']},
        ])
        return optimizer