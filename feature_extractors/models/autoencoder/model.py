import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import os

class AutoEncoder(nn.Module):
    def __init__(self, config, input_shape=(1, 256, 256)):
        super(AutoEncoder, self).__init__()

        self.encoder = self.build_layers(config["model"]["encoder"], conv=True)
        self.decoder = self.build_layers(config["model"]["decoder"], conv=False)
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            feature_map = self.encoder(dummy_input)
            self.feature_map_size = feature_map.shape[1:]
        
        flattened_size = torch.prod(torch.tensor(self.feature_map_size)).item()
        latent_dim = config["model"]["latent_dim"]
        self.fc_enc = nn.Sequential(
            nn.Linear(flattened_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, latent_dim))
        self.fc_dec = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, flattened_size))

    def build_layers(self, layers_config, conv=True):
        layers = []
        for layer in layers_config:
            layers.append(nn.Conv2d(**layer["convolution"]) if conv else nn.ConvTranspose2d(**layer["convolution"]))
            if conv:
                layers.append(nn.BatchNorm2d(layer["convolution"]["out_channels"]))
            
            if layer["activation"] == "relu":
                layers.append(nn.ReLU())
            elif layer["activation"] == "tanh":
                layers.append(nn.Tanh())
            elif layer["activation"] == "selu":
                layers.append(nn.SELU())
            else:
                raise ValueError(f"Unsupported activation function: {layer['activation']}")
                
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        latent = self.fc_enc(x)
        x = self.fc_dec(latent).view(x.size(0), *self.feature_map_size)
        x = self.decoder(x)
        return x, latent

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return self.fc_enc(x)

def get_model(config):
    with open(os.path.join(config["path"], "config.yaml"), "r") as f:
        model_config = yaml.safe_load(f)
        return AutoEncoder(model_config)