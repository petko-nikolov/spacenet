import sys
import yaml
import tqdm
import dataset
import argparse
import os
import cv2
import numpy as np
import torch
from pysemseg.utils import load_model, import_type, prompt_delete_dir
from predict import (
    get_predictions, watershed_transform, find_component_contours,
    get_polygons
)
from dataset import SpacenetOffNadirDataset
from  transforms import SpaceNetTransform
import pickle


if '__main__' == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True,
                        help=('data dir'))
    parser.add_argument('--output-dir', type=str, required=True,
                        help=('Output dir'))
    parser.add_argument('--model', type=str, required=True,
                        help=('model'))
    parser.add_argument('--checkpoint', type=str, required=True,
                        help=('Checkpoint path.'))
    parser.add_argument('--context', type=float, required=False,
                        default=0.8,
                        help='Path to the dataset root dir.')
    parser.add_argument('--use-box-percent', type=float, required=False,
                        default=0.1,
                        help='Path to the dataset root dir.')
    args = parser.parse_args()

    if os.path.exists(args.output_dir):
        prompt_delete_dir(args.output_dir)
    os.makedirs(args.output_dir)

    model_cls = import_type(args.model)

    checkpoint = torch.load(args.checkpoint)

    with open(os.path.join(os.path.dirname(args.checkpoint), 'args.yaml')) as f:
        ckpt_args = yaml.load(f)

    dataset = SpacenetOffNadirDataset(
        args.data_dir, 'train', val_ratio=0.1, **ckpt_args['dataset_args'])

    model = load_model(
        args.checkpoint,
        model_cls,
        SpaceNetTransform,
        device=torch.device('cuda:0'))

    for image_id, image, mask in tqdm.tqdm(dataset):
        predictions, probabilities = get_predictions(model, image)
        watershed_mask = watershed_transform(image, predictions[0])
        contours = find_component_contours(watershed_mask)
        height, width = image.shape[0], image.shape[1]
        preds = (watershed_mask > 1).astype(np.uint8)
        np.random.shuffle(contours)
        contours = contours[:int(args.use_box_percent * len(contours))]
        for i, cnts in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnts[0])
            y1, x1 = (
                int(max(y - h * args.context, 0)),
                int(max(x - w * args.context, 0)),
            )
            y2, x2 = (
                int(min(y + h + h * args.context, height)),
                int(min(x + w + w * args.context, width))
            )
            crop_image = image[y1:y2, x1:x2]
            crop_segment = preds[y1:y2, x1:x2]
            crop_mask = (mask[y1:y2, x1:x2] == 1).astype(np.uint8)

            with open(os.path.join(args.output_dir, image_id + '_box{}'.format(i)), 'wb') as f:
                pickle.dump((crop_image, crop_segment, crop_mask), f)
