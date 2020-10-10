#/usr/bin/env python3
# -*- coding: utf -*-
#
# Copy-Move Forgery Detection Using Multiresolution Local Binary Patterns
#
# Usage example:
# $ mkdir out
# $ python main.py dataset out

import math
import pickle
import sys
from bisect import bisect
from functools import partial
from os import walk, makedirs
from os.path import exists, join
from random import randint
from time import time

import cv2
import numpy as np
from scipy import stats
from scipy.spatial import KDTree
from skimage.feature import local_binary_pattern
from skimage.filters import wiener
from skimage.measure import ransac
from skimage.restoration import unsupervised_wiener
from skimage.transform import ProjectiveTransform
from tqdm import tqdm

# Constants

PSF = np.ones((5, 5)) / 25  # Point Spread Function for Wiener Filter
NEW_RESOLUTION = (260, 260) # Image resolution
BLOCK_SIZE = 10             # A block has a BLOCK_SIZE x BLOCK_SIZE resolution
SPACE = 5                   # Distance between origin of each block
NBEST = 50                  # Number of closest neighbors to choose after
                            # sorting
DIST = 50                   # Minimum euclidean distance between possible 
                            # matches

def show(image):
    cv2.imshow('image', image)
    cv2.waitKey(-1)

def normalized_lbp_histogram(lbp_image, P, method='default'):
    ''' Create histogram for image.
            method - LBP method
    '''
    n_lbps = {'default':2**P, 'nri_uniform':P*(P-1)+3, 'uniform':P+1, 'ror':2**P}
    hist = np.histogram(lbp_image, bins=range(n_lbps[method]))[0]
    sum_ = np.sum(hist)
    return hist / (sum_ + 1e-20)
    # FIXME: Check vector size
    # if method in ['default', 'nri_uniform', 'ror', 'uniform']:
    #     lbp_image_new = np.array(lbp_image, copy=True).astype(dtype='uint8')
    #     lines, cols = lbp_image_new.shape
    #     n_pixels = lines*cols
    #     vec = []
    #     for t in range(n_lbps[name]):
    #         vec.append(np.sum(lbp_image_new == t) / n_pixels)

    #     # Unit length
    #     sum_ = sum(vec)
    #     for i in range(len(vec)):
    #         vec[i] = vec[i] / (sum_ + 1e-10)
    #     return vec
    # else:
    #     raise ValueError(f'No implementation for {method}')

def preprocess(image, use_wiener=False, psf=PSF, resolution=NEW_RESOLUTION):
    '''Convert to grayscale, apply wiener filter and resize'''

    # Grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Wiener filter
    if use_wiener:
        image, _ = unsupervised_wiener(image.astype(dtype='float64')/255.0, PSF)
        image = image - image.min()
        image = ((image/image.max())*255).astype(dtype='uint8')

    # Resize
    image = cv2.resize(image, NEW_RESOLUTION)
    return image

def load_images(director, file_ending='_t.bmp'):
    ''' Load images from folder (and subfolders)'''
    images = []
    for root, folders, files in walk(director):
        files = [f for f in files if f.endswith(file_ending)]
        images.extend([(join(root, f),
                        preprocess(cv2.imread(join(root, f))),
                        cv2.resize(cv2.imread(join(root, f)[:-4] + '_mask.bmp'),
                                   NEW_RESOLUTION)[:,:,0]) for f in files])
    return images

def extract_blocks(lbp_image, B, space, P=8, method='default', image_mask=False):
    vectors_and_pos = []
    n_lines, n_cols = lbp_image.shape
    max_i = n_lines - B
    max_j = n_cols - B
    for i in tqdm(range(0, max_i, space)):
        for j in range(0, max_j, space):
            pos = (i, j)
            block = lbp_image[i:i + B, j:j + B]
            vector = normalized_lbp_histogram(block, P, method=method)
            vectors_and_pos.append((list(vector), pos))
    vectors, positions = list(zip(*sorted(vectors_and_pos, key=lambda x: x[0])))
    return vectors, positions

def debug_sorted_blocks(lbp_image, B, name, image_mask, vectors, positions):
    print('Debugging')
    image_sorted_blocks = np.zeros((B, len(vectors) * (B + 1), 3), dtype='uint8')
    for n_blocks, (i,j) in enumerate(positions):
        # print(n_blocks)
        # print(0, B, B*n_blocks+1, B*n_blocks+1+B)
        image_sorted_blocks[0:B, B*n_blocks+1:B*n_blocks+B+1, 0] = lbp_image[i:i+B, j:j+B]
        image_sorted_blocks[0:B, B*n_blocks+1:B*n_blocks+B+1, 1] = lbp_image[i:i+B, j:j+B]
        print(f'{(i+B//2, j+B//2)} = {image_mask[i+B//2, j+B//2] == 255}')
        if image_mask[i+B//2, j+B//2] == 255:
            image_sorted_blocks[0:B, B * n_blocks + 1:B * n_blocks + B + 1, 2] = 0
        else:
            image_sorted_blocks[0:B, B*n_blocks+1:B*n_blocks+B+1, 2] = lbp_image[i:i+B, j:j+B]
        image_sorted_blocks[:, n_blocks*B, 0] = 0
        image_sorted_blocks[:, n_blocks*B, 1] = 255
        image_sorted_blocks[:, n_blocks*B, 2] = 255
        # print(image_sorted_blocks[0:B, B*n_blocks:B*n_blocks+B+2, 0])
        # cv2.imwrite('/home/thiago/repos/cmfd/block.bmp', image_sorted_blocks[0:B, B*n_blocks:B*n_blocks+B+2])
        # cv2.imshow('block.bmp', image_sorted_blocks[0:B, B*n_blocks:B*n_blocks+B+2, 0])
        # cv2.waitKey(-1)
    cv2.imshow('mask', image_mask)
    cv2.waitKey()
    cv2.imwrite(f'/home/thiago/repos/cmfd/blocks_{name}.bmp', image_sorted_blocks)

def eucl_dist(v1, v2):
    '''Squared distance between two vectors'''
    return math.sqrt(sum([(c1-c2)**2 for c1, c2 in zip(v1, v2)]))

def find_matching_blocks(vectors, positions, nbest=50, dist=50):
    closest_vectors = []
    closest_positions = []
    blocks_idxs = []
    for i, (v1, p1) in tqdm(list(enumerate(zip(vectors, positions)))):

        # Brute force (very slow!)
        # s = sorted(zip(vectors, positions, range(len(vectors))), key=lambda x: eucl_dist(v1, x[0]))
        # for vector, position, idx in s:
        #     if eucl_dist(p1, position) < dist:
        #         closest_idx = idx
        #         break

        candidates = {'vectors': [], 'positions': []}
        
        idx_diff = 1
        sign = 1 if i != len(vectors) - 1 else -1
        change_sign = True
        n_searched = 0
        while len(candidates['vectors']) < nbest:
            n_searched += 1
            i_next = i + sign * idx_diff
            i_last = len(positions) - 1

            # Change idx_diff every two steps
            if n_searched%2 == 0 or not change_sign:
                idx_diff += 1
            
            # If it is the end, stop changing sign (go only one way)
            if change_sign and i_next < 0 or i_next > i_last:
                change_sign = False
                sign *= -1
                continue
            # Going in only one direction and got to the end
            if not change_sign and i_next < 0 or i_next > i_last:
                print('breaking')
                break


            p2 = positions[i_next]
            v2 = vectors[i_next]
            if eucl_dist(p1, p2) >= dist:
                candidates['vectors'].append(v2)
                candidates['positions'].append(p2)

            # Change sign
            if change_sign:
                sign *= -1

        tree = KDTree(np.array(candidates['vectors']))
        closest_idx = vectors.index(candidates['vectors'][tree.query(v1)[1]])
        closest_vectors.append(vectors[closest_idx])
        closest_positions.append(positions[closest_idx])
        blocks_idxs.append(closest_idx)
    return closest_vectors, closest_positions, blocks_idxs

def consensus(blocks_idxs):
    matches = []
    for i, votes in enumerate(zip(*blocks_idxs)):
        [i_closest], [frequency] = stats.mode(votes)
        if frequency > len(votes) // 2:
            matches.append((i, i_closest))
    return matches

def evaluation_metrics(image_mask, matches_positions, block_size=BLOCK_SIZE):
    predicted_mask = np.zeros(image_mask.shape, dtype='uint8')
    for (i1, j1), (i2, j2) in matches_positions:
        predicted_mask[i1:i1 + block_size, j1:j1 + block_size] = 255
        predicted_mask[i2:i2 + block_size, j2:j2 + block_size] = 255
    
    mask_area = np.sum(image_mask == 255)
    pred_mask_area = np.sum(predicted_mask == 255)
    
    overlapping = np.sum((image_mask & predicted_mask) == 255)
    recall = overlapping / mask_area
    precision = overlapping/pred_mask_area
    # cv2.imshow('original_mask', image_mask)
    # cv2.imshow('predicted_mask', predicted_mask)
    # cv2.imshow('overlapping', image_mask & predicted_mask)
    # cv2.waitKey(-1)
    # cv2.imshow('overlapping', (np.logical_and(image_mask, predicted_mask)).astype(dtype='uint8'))
    print((f'Recall: {overlapping}/{mask_area}={100*overlapping/mask_area:.2f}% | '
           f'Precision: {overlapping}/{pred_mask_area}={100*overlapping/pred_mask_area:.2f}%'))
    return precision, recall

def remove_outliers_ransac(matches):
    new_matches = []
    model, inliers = ransac(matches,
                            ProjectiveTransform, min_samples=8,
                            residual_threshold=1, max_trials=5000)
    for [i] in np.argwhere(inliers):
        new_matches.append((tuple(matches[0][i]), tuple(matches[1][i])))
    return new_matches


if __name__ == '__main__':
    # Create output directory
    out_dir = sys.argv[2]
    if not exists(out_dir):
        print(f'Creating output directories: {out_dir}')
        makedirs(out_dir)

    print('Parameters:')
    print(f'PSF: {PSF}\n'
          f'NEW_RESOLUTION: {NEW_RESOLUTION[0]} x {NEW_RESOLUTION[1]}\n'
          f'BLOCK_SIZE: {BLOCK_SIZE}\n'
          f'SPACE: {SPACE}\n'
          f'NBEST: {NBEST}\n'
          f'DIST: {DIST}')

    precisions = []
    recalls = []
    for path, image, image_mask in load_images(sys.argv[1]):
        print(f'\nImage: {path}')

        # Create color image to display
        color_img = np.zeros((image.shape[0], image.shape[1], 3), dtype='uint8')
        color_img[:, :, 0] = image
        color_img[:, :, 1] = image
        color_img[:, :, 2] = image

        # Create LBP images
        methods = [('nri_uniform1','nri_uniform', 8, 1),
                   ('nri_uniform2', 'nri_uniform', 12, 2),
                   ('ror', 'ror', 8, 1),
                   ('uniform', 'uniform', 16, 2)]
        images_lbp = {}
        for name, method, P, R in methods:
            images_lbp[name] = local_binary_pattern(image, P, R, method)

        # Create blocks (represented by feature vectors and positions)
        vectors = {}
        positions = []
        blocks_dir = join(out_dir, 'blocks')
        if not exists(blocks_dir):
            print(f'Creating blocks directory: {blocks_dir}')
            makedirs(blocks_dir)
        blocks_filename = path.replace('/', '_') + '.blocks'
        blocks_path = join(blocks_dir, blocks_filename)
        if exists(blocks_path):
            print(f'Reading from disk: {blocks_path}')
            vectors, positions = pickle.load(open(blocks_path, 'rb'))
        else:
            t_start = time()
            for name, method, P, R in methods:
                print(f'Computing blocks for method: {name}')
                vectors[name], positions = extract_blocks(
                    images_lbp[name],
                    BLOCK_SIZE,
                    SPACE,
                    method=method)
            print(f'Time to generate blocks: {time() - t_start}s')
            print(f'Writing blocks to: {blocks_path}')
            pickle.dump((vectors, positions), open(blocks_path, 'wb'))

        # # For debugging
        # for name, method, _, _ in methods:
        #     debug_sorted_blocks(images_lbp[name], BLOCK_SIZE, name, image_mask, vectors[name], positions)

        # Block matching (using lexicographical ordering and kdtree)
        t_start = time()
        closest_vectors_list = []
        closest_positions_list = []
        blocks_idxs_list = []
        for name, method, _, _ in methods:
            print(f'Matching blocks for method {name}')
            closest_vectors, closest_positions, blocks_idxs = find_matching_blocks(vectors[name], positions)
            closest_vectors_list.append(closest_vectors)
            closest_positions_list.append(closest_positions)
            blocks_idxs_list.append(blocks_idxs)
        matches = consensus(blocks_idxs_list)
        print(f'time (match blocks): {time() - t_start}s')

        positions_1 = np.array([positions[i] for i, _ in matches])
        positions_2 = np.array([positions[j] for _, j in matches])

        # RANSAC
        matches_positions = np.array([(positions[i], positions[j]) for i, j in matches])
        # matches_positions = remove_outliers_ransac((positions_1, positions_2))

        # Calculate evaluation metrics
        precision, recall = evaluation_metrics(image_mask, matches_positions)
        precisions.append(precision)
        recalls.append(recall)

        # Show results
        for (i1, j1), (i2, j2) in matches_positions:
            i1, j1, i2, j2 = [c + BLOCK_SIZE // 2 for c in (i1, j1, i2, j2)]
            cv2.line(color_img, (j1, i1),
                                (j2, i2), (255, 0, 0), 1)
            cv2.circle(color_img, (j1, i1), 2, (0,0,255), 1)
            cv2.circle(color_img, (j2, i2), 2, (0, 255, 0), 1)
            # show(color_img)

        print(f'Writing to: {out_dir + "/" + path.replace("/", "_") + ".jpg"}')
        cv2.imwrite(sys.argv[2] + '/' + path.replace("/", "_") + ".jpg", color_img)

        # Write LBP Images
        lbp_dir = join(out_dir, 'lbp_images')
        if not exists(lbp_dir):
            print(f'Creating lbp_images directory: {lbp_dir}')
            makedirs(lbp_dir)
        for name, method, _, _ in methods:
            lbp_image_file = join(lbp_dir, path.replace("/", "_") + f'_{name}_.jpg')
            print(f'Writing to: {lbp_image_file}')
            cv2.imwrite(lbp_image_file, images_lbp[name])

    print(f'Precisions: {precisions}')
    print(f'Recalls: {recalls}')
    print(f'Avg precision: {np.average(precisions)}')
    print(f'Avg recall: {np.average(recalls)}')
