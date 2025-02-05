import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import statistics
import pandas as pd

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, create_pairwise_gaussian

def noiseReduction(images_label_gray_general, label_img_names, label_dir, result_path, gt_prob):
    
    GT_PROB = gt_prob

    labeled_dir = label_dir

    result_test_path_general = result_path
    
    crf_file_names_general = label_img_names
    
    preCRF = images_label_gray_general

    postCRF = []

    images_label_gray = np.array(images_label_gray_general)
    #labels = images_label_gray[5]
    for image_label_gray in images_label_gray:

        H, W = image_label_gray.shape
        image_label_gray = image_label_gray.flatten()


        invalid_indices = np.where((image_label_gray < 0) | (image_label_gray > 10))

        # Check if there are any invalid indices
        if invalid_indices[0].size > 0:
            print("Invalid label indices:", invalid_indices[0])  # Print the flat indices of invalid labels
            print("Invalid label values:", image_label_gray[invalid_indices])  # Print the invalid label values
        else:
            print("All labels are within the valid range (0 to 10).")


        # Create a dense CRF model
        num_classes = 11  # 0 to 10, so 11 classes
        d = dcrf.DenseCRF2D(W, H, num_classes)  # (width, height, num_classes)

        # Create unary potentials
        unary = unary_from_labels(image_label_gray, n_labels = num_classes, gt_prob = GT_PROB, zero_unsure = False)
        d.setUnaryEnergy(unary)

        # Add pairwise terms (spatial & appearance smoothness)
        d.addPairwiseGaussian(sxy=3, compat=10)

        # Inference
        Q = d.inference(5)  # Perform 5 CRF iterations

        # Convert Q into a label image
        refined_label = np.argmax(Q, axis=0)  # Most probable class for each pixel

        # Reshape back to the original image shape
        refined_label = refined_label.reshape((H, W))
        postCRF.append(np.array(refined_label))

    # Saving the crf enhanced images into crf dir

    # Create the crftestdir path
    crftestdir = result_test_path_general + "/crf"
    # Convert to Path object for easier manipulation
    crftestdir_path = Path(crftestdir)

    # Create the crf directory if it does not exist
    if not crftestdir_path.exists():
        crftestdir_path.mkdir(parents=True, exist_ok=True)
    refined_labels_case = postCRF
    crf_file_names_case = crf_file_names_general
    # Convert the array to uint8 type if necessary
    for index, refined_label in enumerate(refined_labels_case):
        refined_label_uint8 = refined_label.astype(np.uint8)
        # Create a PIL image from the NumPy array
        image = Image.fromarray(refined_label_uint8)
        # Save the crf refined images to the specified crf folder
        image.save(crftestdir_path / (crf_file_names_case[index] + ".png"))  # Use / for path concatenation
    return postCRF