from lightly.loss import NTXentLoss
from lightly.models.modules.heads import SimCLRProjectionHead


class SimCLRModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        
        self.cfg = cfg

        if cfg['encoder_arch'] == "resnet18":
            resnet = torchvision.models.resnet18(pretrained=False)
            resnet.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            raise ValueError(f"Unsupported encoder architecture: {cfg['encoder_arch']}")
    
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        hidden_dim = resnet.fc.in_features
        self.projection_head = SimCLRProjectionHead(hidden_dim, hidden_dim, cfg['projection_dim'])

        self.criterion = NTXentLoss(temperature=cfg['temperature'])

    def forward(self, x):
        h = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(h)
        return z

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log("train_loss_ssl", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)

        alignment = (z0 - z1).norm(dim=1).mean()

        z0_norm = nn.functional.normalize(z0, dim=1)
        pairwise_dist = torch.cdist(z0_norm, z0_norm, p=2)
        uniformity = torch.log(torch.exp(-2 * pairwise_dist ** 2).mean())

        self.log("val_alignment", alignment, prog_bar=True)
        self.log("val_uniformity", uniformity, prog_bar=True)

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(), lr=cfg['lr'], momentum=cfg['momentum'], weight_decay=cfg['weight_decay']
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]