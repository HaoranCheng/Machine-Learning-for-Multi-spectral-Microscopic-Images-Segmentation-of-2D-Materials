import numpy as np
import cv2


def CCA_postprocessing(pred_array):
    min_size = 25
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(pred_array, connectivity=8)
    sizes = stats[1:, -1];
    nb_components = nb_components - 1
    for n in range(0, nb_components):
        if sizes[n] < min_size:
            pred_array[output == n + 1] = 0
    return pred_array