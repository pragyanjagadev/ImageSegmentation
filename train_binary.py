import os
import sys
from dataset_binary import DirDataset
from unet_binary import UnetBinary
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor

# Define hyperparameters
in_channels = 1
out_channels = 1
learning_rate = 0.001
batch_size = 32
epochs = 10
dataset = 'IAM'

# Initialize Lightning module
model = UnetBinary(in_channels=1)

dataset = DirDataset(f'./dataset/{dataset}/augmented_data', f'./dataset/{dataset}/masks')
n_val = int(len(dataset) * 0.1)
n_train = len(dataset) - n_val

train_ds, val_ds = random_split(dataset, [n_train, n_val])
train_loader = DataLoader(train_ds, batch_size=1, pin_memory=True, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=1, pin_memory=True, shuffle=False)

try:
    log_dir = sorted(os.listdir('lightning_logs_binary'))[-1]
except IndexError:
    log_dir = os.path.join('lightning_logs_binary', 'version_0')

# Initialize a Lightning Trainer and fit the model
checkpoint_callback = ModelCheckpoint(
    dirpath=log_dir,
    filename="{epoch}_{val_loss:.2f}",  # 'checkpoints',
    save_top_k=3,
    monitor="val_loss",
    mode="min",
    verbose=True,
)

stop_callback = EarlyStopping(
    monitor='val_loss',
    mode='min',  # 'auto',
    patience=3,  # 5,
    verbose=True,
)
learning_rate_monitor = LearningRateMonitor(
    logging_interval="step"
)

trainer = Trainer(
    #fast_dev_run=True,
    max_epochs=1,
    callbacks=[checkpoint_callback, stop_callback, learning_rate_monitor],
)
trainer.fit(model, train_loader, val_loader)
