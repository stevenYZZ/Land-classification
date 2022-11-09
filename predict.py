import argparse
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from models import UNet
# from utils import plot_img_and_mask

from tqdm import tqdm


# input_dir = Path("./data_all_256")
# output_dir = Path("./data_all_256_rotate")
input_path = Path("./data_all_256_origin/data")
output_path = Path("./data_all_256_origin/predict_datasetRotate")
model_path = Path('./checkpoints/rotateSplit_b10/checkpoint_epoch100.pth')



def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor, is_label=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)[0]
        else:
            probs = torch.sigmoid(output)[0]

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((full_img.size[1], full_img.size[0])),
            transforms.ToTensor()
        ])

        full_mask = tf(probs.cpu()).squeeze()

    if net.n_classes == 1:
        return (full_mask > out_threshold).numpy()
    else:
        return F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    # parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images')
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')

    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))


if __name__ == '__main__':
    args = get_args()
    # in_files = args.input
    # out_files = get_output_filenames(args)
    in_files = input_path
    out_files = output_path
    model = model_path
    if not os.path.exists(out_files):
        os.makedirs(output_path)


    net = UNet(n_channels=3, n_classes=2, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    net.load_state_dict(torch.load(model, map_location=device))

    logging.info('Model loaded!')
    img_num = len(os.listdir(input_path))
    count = 0
    # with tqdm(total = os.listdir(input_path), desc = f'Epoch{epoch} / {img_num}', unit = 'img') as pbar:
    with tqdm(total = img_num) as pbar:
        for filename in os.listdir(input_path):
            pbar.update(1)
            input_file = input_path/filename
            output_file = output_path/filename


            logging.info(f'\nPredicting image {filename} ...')

            img = Image.open(input_file)

            mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)
            # print(mask.shape)
            # print(mask.shape[0])
            # print(np.argmax(mask, axis=0).shape)
            # exit()
            # result = mask_to_image(mask)
            result = Image.fromarray((np.argmax(mask, axis=0) / mask.shape[0]).astype(bool))
            result.save(output_file)
            logging.info(f'Mask saved to {output_file}')


            # if not args.no_save:
            # # out_filename = out_files[i]


        # if args.viz:
        #     logging.info(f'Visualizing results for image {filename}, close to continue...')
        #     plot_img_and_mask(img, mask)
