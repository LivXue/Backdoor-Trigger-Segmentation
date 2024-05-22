'''Define some commonly used metric for evaluation'''

import numpy as np

'''  metrics for evaluation '''
def calculate_precision(pred_mask, true_mask):
    '''Compute the precision of detection results'''
    true_positive = np.sum(np.logical_and(pred_mask == 1, true_mask == 1))
    false_positive = np.sum(np.logical_and(pred_mask == 1, true_mask == 0))

    if true_positive + false_positive == 0:
        return 0

    precision = true_positive / (true_positive + false_positive)
    return precision


def calculate_recall(pred_mask, true_mask):
    '''Compute the recall of detection results'''
    true_positive = np.sum(np.logical_and(pred_mask == 1, true_mask == 1))
    false_negative = np.sum(np.logical_and(pred_mask == 0, true_mask == 1))
    recall = true_positive / (true_positive + false_negative)
    return recall


def calculate_iou(pred_mask, true_mask):
    '''Compute the iou of detection results'''
    intersection = np.logical_and(pred_mask, true_mask)
    union = np.logical_or(pred_mask, true_mask)
    iou = np.sum(intersection) / np.sum(union)
    return iou
