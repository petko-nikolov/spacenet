import itertools
from collections import defaultdict
import numpy as np
from geomet import wkt
import torch
import torch.nn.functional as F
import cv2
import tqdm

from pysemseg import transforms

def watershed_transform(image, predictions):
    kernel = np.ones((3, 3), dtype=np.uint8)
    opening = cv2.morphologyEx(
        (predictions == 1).astype(np.uint8), cv2.MORPH_OPEN,
        kernel, iterations=3)
    closing = cv2.morphologyEx(
        opening.astype(np.uint8), cv2.MORPH_CLOSE,
       kernel, iterations=3)
    _, markers = cv2.connectedComponents(closing)
    markers = markers + 1
    unknown = cv2.dilate(
        (predictions == 2).astype(np.uint8),
        kernel=np.ones((3, 3), dtype=np.uint8), iterations=3)
    unknown = np.where(closing, 0, unknown)
    markers[unknown==1] = 0
    watershed = cv2.watershed(
        (image[:, :, :3] * 255).astype(np.uint8), markers)
    return watershed


def group_contours(contours, hierarchy):
    grouped_contours = defaultdict(dict)
    for i in range(hierarchy.shape[1]):
        epsilon = 0.01 * cv2.arcLength(contours[i], True)
        contour = cv2.approxPolyDP(contours[i], epsilon, closed=True)
        if hierarchy[0, i, 3] == -1:
            grouped_contours[i]['external'] = contour
        else:
            if 'internals' not in grouped_contours[hierarchy[0, i, 3]]:
                grouped_contours[hierarchy[0, i, 3]]['internals'] = []
            grouped_contours[hierarchy[0, i, 3]]['internals'].append(contour)
    return grouped_contours


def find_component_contours(components):
    max_component = components.max()
    contours = []
    for i in range(2, max_component + 1):
        poly_mask = (components == i).astype(np.uint8)
        poly_mask = cv2.dilate(poly_mask, np.ones((3,3), dtype=np.uint8), iterations=2)
        _, cnts, hierarchy = cv2.findContours(
            poly_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        groups = group_contours(cnts, hierarchy)
        for group in groups.values():
            contours.append([group['external']] + group.get('internal', []))
    return contours

def get_predictions(model, image):
    logits = model(image)
    probabilities = torch.nn.functional.softmax(logits.to('cpu:0'), dim=1)
    probabilities = probabilities.data.to('cpu:0').numpy()
    probabilities = probabilities.transpose(0, 2, 3, 1)
    predictions = probabilities.argmax(axis=3)
    return predictions, probabilities

def get_polygons(contours):
    polygons = []
    for cnts in contours:
        polys = [cnt[:, 0, :,].tolist() for cnt in cnts]
        for poly in polys:
            if poly[0] != poly[-1]:
                poly.append(poly[0])
        polygons.append(polys)
    return polygons

def get_solution_rows(image_id, polygons):
    buildings = []
    for i, poly in enumerate(polygons):
        buildings.append(
            '{}, {}, {}, 1.0'.format(
                image_id,
                i,
                wkt.dumps({'type': 'Polygon', 'coordinates': poly}, decimals=2)
            )
        )
    return buildings

def get_predictions_msf(model, image, scales=[0.8, 0.9, 1.0, 1.1, 1.2]):
    height, width = image.shape[:2]
    probabilities = []
    for flip, scale in itertools.product([False, True], scales):
        hs, ws = int(height * scale), int(width * scale)
        rimage = transforms.Resize((hs, ws))(image)
        if flip:
            rimage = cv2.flip(rimage, 1)
        logits = model(rimage)
        logits = F.interpolate(logits, size=[height, width], mode='bilinear', align_corners=True)
        if flip:
            logits = torch.flip(logits, dims=[3])
        probs = torch.nn.functional.softmax(logits.to('cpu:0'), dim=1)
        probs = probs.data.to('cpu:0').numpy()
        probs = probs.transpose(0, 2, 3, 1)
        probabilities.append(probs)
    probabilities = sum(probabilities) / len(probabilities)
    predictions = probabilities.argmax(axis=3)
    return predictions, probabilities


if __name__ == '__main__':
    test_dataset = SpacenetOffNadirDataset(
        '/mnt/hdd/datasets/spacenet/sample/', 'val', val_ratio=1.0, **args['dataset_args'])
    solution_rows = []
    for image_id, image, mask in tqdm.tqdm(test_dataset):
        predictions, probabilities = get_predictions(model, image)
        watershed_mask = watershed_transform(image, predictions[0])
        contours = find_component_contours(watershed_mask)
        polygons = get_polygons(contours)
        solution_rows.extend(get_solution_rows(image_id, polygons))
    with open('/home/petko/sample_solution.csv', 'w') as f:
        for row in solution_rows:
            f.write(row + '\n')
