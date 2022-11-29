"""
Written by KrishPro @ KP

filename: `train.py`
"""

try:
    from model import ViT
    from data import DataModule
except ImportError:
    from caption_images.model import ViT
    from caption_images.data import DataModule

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple
from pytorch_lightning import LightningModule, Trainer

class Model(LightningModule):
    def __init__(self, d_model:int, n_heads:int, dim_feedforward:int, num_layers:int, tgt_vocab_size:int, log_interval:bool=None, learning_rate:float=3e-4, label_smoothing:float=0.1, dropout_p:float=0.1, pad_idx:int=0) -> None:
        self.save_hyperparameters()
        super().__init__()

        self.criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=label_smoothing)
        self.tgt_vocab_size = tgt_vocab_size
        self.learning_rate = learning_rate
        self.log_interval = log_interval

        self.ViT = ViT(d_model=d_model, n_heads=n_heads, dim_feedforward=dim_feedforward, num_layers=num_layers, tgt_vocab_size=tgt_vocab_size, dropout_p=dropout_p, pad_idx=pad_idx)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx):
        images, captions = batch
        
        generated_captions = self.ViT(images, captions[:, :-1])

        loss: torch.Tesnor = self.criterion(generated_captions.reshape([-1, self.tgt_vocab_size]), captions[:, 1:].reshape(-1))

        self.log("loss", loss.detach())

        if self.log_interval and (batch_idx % self.log_interval == 0): print(f"Epoch #{self.current_epoch} | Batch #{batch_idx} | Loss: {loss.detach()}")
        return loss
    

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx):
        images, captions = batch

        generated_captions = self.ViT(images, captions[:, :-1])

        loss = self.criterion(generated_captions.reshape([-1, self.tgt_vocab_size]), captions[:, 1:].reshape(-1))

        self.log("val_loss", loss.detach(), prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)
    
config = {
    "dims": {
         "d_model": 256,
         "n_heads": 4,
         "dim_feedforward": 1024,
         "num_layers": 1,
         "tgt_vocab_size": 30_000,
         "pad_idx": 0,
         "dropout_p": 0.1,
    },
    "learning_rate": 1e-4,
    "label_smoothing": 0.1,
    "images_dir": "/home/krish/Datasets/flickr30k/images",
    "data_path": "data/data.csv",
    "batch_size": 32,
    "val_ratio": 0.1,
    "log_interval": None,
    "use_workers": True,
    "trainer": {
        "accelerator": "gpu",
        "devices": 1,
        "precision": 32,
        "accumulate_grad_batches": 1,
        "overfit_batches": 1,
        "max_epochs": 50
    }

}

def train(config):
    model = Model(learning_rate=config['learning_rate'], label_smoothing=config['label_smoothing'], log_interval=config['log_interval'], **config['dims'])

    datamodule = DataModule(config['images_dir'], config['data_path'], batch_size=config['batch_size'], val_ratio=config['val_ratio'], use_workers=config['use_workers'])

    trainer = Trainer(**config['trainer'])

    trainer.fit(model, datamodule)


if __name__ == '__main__':
    train(config)