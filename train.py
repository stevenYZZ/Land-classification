import argparse
import logging
import os
from pathlib import Path


import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import UNet
from utils.data_loading import BasicDataset
from utils.dice_score import dice_loss
from evaluate import evaluate

# dir_dataset = Path('./data')
dir_dataset = Path('../building-map-generator-datasets/data_all_256_rotate/')

dir_data = dir_dataset/"data/"
dir_label = dir_dataset/"label/"

dir_output = Path('./checkpoints/')
run_name = "try"

dir_checkpoint = dir_output/run_name

args_classes = 2
args_batch_size = 20
args_epochs = 100
args_lr = 1e-5
args_scale = 0.5


def train_net(net,
              device,
              epochs: int = 5,
              batch_size: int = 1,
              learning_rate: float = 1e-5,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              img_scale: float = 0.5,
              amp: bool = False):

    # 1. Create dataset

    train_dataset = BasicDataset(dir_data/"train", dir_label/"train", img_scale)
    val_dataset = BasicDataset(dir_data/"val", dir_label/"val", img_scale)
    # print(train_dataset.__getitem__(0)['label'].shape)  # torch.Size([256, 256])
    # print(train_dataset.__getitem__(0)['data'].shape)   # torch.Size([256, 256, 3])
    # print(train_dataset.__getitem__(0)['label'].dtype)  # int64
    # print(train_dataset.__getitem__(0)['data'].dtype)   # float32
    # print(train_dataset.__getitem__(0)['data'])   # torch.Size([256, 256, 3])
    # print(train_dataset.__getitem__(0)['data'].sum())   # torch.Size([256, 256, 3])
    # print(train_dataset.__getitem__(0)['label'])   # torch.Size([256, 256, 3])
    # exit()
    n_train = len(train_dataset)
    n_val = len(val_dataset)

    # 2. Create data loaders
    
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=True, **loader_args)

    # for batch in train_loader:
    #     images = batch['data']
    #     true_masks = batch['label']

    # (Initialize logging)
    experiment = wandb.init(project='BUILDING-MAP-GENERATOR', resume='allow', anonymous='never', name = run_name)
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                  val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
                                  amp=amp))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    # optimizer = optim.Adam(net.parameters(), lr = learning_rate, weight_decay=1e-8)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)    #自动混合精度
    criterion = nn.CrossEntropyLoss()
    global_step = 0

    # 5. Begin training

    # 遍历epoch
    for epoch in range(1, epochs + 1):
        net.train()
        epoch_loss = 0
        # 加载进度条
        with tqdm(total = n_train, desc = f'Epoch{epoch} / {epochs}', unit = 'img') as pbar:
            # 遍历batch
            for batch in train_loader:
                images = batch['data']
                true_masks = batch['label']
                # print(images.shape)   #(1, 256, 256, 3)
                assert images.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device = device, dtype = torch.float32)
                true_masks = true_masks.to(device = device, dtype = torch.long)

                with torch.cuda.amp.autocast(enabled = amp):
                    mask_pred = net(images)
                    # print(mask_pred.shape)
                    # print(mask_pred.grad)
                    # print(mask_pred.sum(dim = 0))
                    # print(true_masks.shape)
                    # print(true_masks.sum(dim = 0))
                    # exit()
                    loss = criterion(mask_pred, true_masks) \
                           + dice_loss(F.softmax(mask_pred, dim = 1).float(),
                                       F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                                       multiclass = True)

                optimizer.zero_grad(set_to_none = True) #放在grad_scaler后面
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss' : loss.item(),
                    'step' : global_step,
                    'epoch' : epoch
                })
                pbar.set_postfix({'loss (batch)': loss.item()})

                # Evaluation round
                # 评估某一张图？
                division_step = (n_train // (10 * batch_size))
                
                if division_step > 0:
                    if global_step % division_step == 0:
                        # histograms = {}
                        # for tag, value in net.named_parameters():
                        #     tag = tag.replace('/', '.')
                        #     histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                        #     histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate(net, val_loader, device)
                        scheduler.step(val_score)
                        logging.info(f'Validation Dice score: {val_score}')
                        # print(tag)
                        # print(histograms['Weights/' + tag])
                        # print(histograms['Gradients/' + tag])
                        print(mask_pred.shape)
                        print(true_masks.shape)
                        exit()
                        experiment.log({
                            'learning rate': optimizer.param_groups[0]['lr'],
                            'validation Dice': val_score,
                            'images': wandb.Image(images[0].cpu()),
                            'masks': {
                                'true': wandb.Image(true_masks[0].float().cpu()),
                                'pred': wandb.Image(mask_pred.argmax(dim=1)[0].float().cpu()),
                            },
                            'step': global_step,
                            'epoch': epoch,
                            # **histograms
                        })
        if save_checkpoint:
            if epochs < 10:
                Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                torch.save(net.state_dict(), str(dir_checkpoint / f'checkpoint_epoch{epoch}.pth'))
                logging.info(f'Checkpoint {epoch} saved!')
            elif epoch % 10 == 0:
                Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                torch.save(net.state_dict(), str(dir_checkpoint / f'checkpoint_epoch{epoch}.pth'))
                logging.info(f'Checkpoint {epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    # parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    # parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        # help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=True, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    return parser.parse_args()

if __name__ == '__main__':
    if(not os.path.exists(dir_checkpoint)):
        os.mkdir(dir_checkpoint)

    assert os.path.exists(dir_checkpoint), "checkpoint dir missing"
    args = get_args()

    # 更新超参数   
    args.classes = args_classes
    args.batch_size = args_batch_size
    args.epochs = args_epochs
    args.lr = args_lr
    args.scale = args_scale

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    # 添加预训练模型

    # if args.load:
    #     net.load_state_dict(torch.load(args.load, map_location=device))
    #     logging.info(f'Model loaded from {args.load}')

    net.to(device=device)

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device,
                  img_scale=args.scale,
                #   val_percent=args.val / 100,
                  amp=args.amp)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        raise
