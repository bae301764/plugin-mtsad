from typing import List
import numpy as np
from scipy.stats import iqr, rankdata
from typing import List
from sklearn.metrics import roc_curve, auc, average_precision_score
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def calculate_and_smooth_error_scores(predictions: list or np.ndarray,
                                      ground_truth: list or np.ndarray,
                                      smoothing_window: int = 3,
                                      epsilon:float = 1e-2) -> np.ndarray:
    """
    Calculate and smooth the error scores between test predictions and ground truths.

    Args:
        predictions (list or np.ndarray): The predicted values on the test set.
        ground_truth (list or np.ndarray): The actual ground truth values of the test set.
        smoothing_window (int): The number of elements to consider for smoothing the error scores.
        epsilon (float): A small constant added for numerical stability.

    Returns:
        numpy.ndarray: Smoothed error scores.
    """
    test_delta = np.abs(np.array(predictions) - np.array(ground_truth))
    err_median = np.median(test_delta)
    err_iqr = iqr(test_delta)
    normalized_err_scores = (test_delta - err_median) / (np.abs(err_iqr) + epsilon)
    # smoothe the error scores by a moving average
    smoothed_err_scores = np.zeros_like(normalized_err_scores)
    for idx in range(smoothing_window, len(normalized_err_scores)):
        smoothed_err_scores[idx] = np.mean(normalized_err_scores[idx - smoothing_window: idx])
    return smoothed_err_scores

def calculate_nodewise_error_scores(predictions:np.ndarray,
                                    ground_truth:np.ndarray,
                                    smoothing_window:int=3,
                                    epsilon:float=1e-2) -> np.ndarray:
    # predictions: [total_time_len, num_nodes]
    # ground_truth: [total_time_len, num_nodes]
    # return: [num_nodes, total_time_len - smoothing_window + 1]
    nodewise_error_scores = []
    number_nodes = predictions.shape[1]
    for i in range(number_nodes):
        pred = predictions[:, i]
        gt = ground_truth[:, i]
        scores = calculate_and_smooth_error_scores(pred, gt,
                                                   smoothing_window,
                                                   epsilon)
        nodewise_error_scores.append(scores)
    
    # [num_nodes, total_time_len - smoothing_window + 1]
    return np.stack(nodewise_error_scores, axis=0) 
    

def test_performence(test_result:List[np.ndarray],
                     smoothing_window:int=3,
                     epsilon:float=1e-2) -> tuple:
    # Compute test-set precision, recall, and F1 by combining base-model and attention-based scores 
    # and selecting the optimal threshold on the test data.

    test_predictions, test_ground_truth, test_anomaly_label, test_attention_score = test_result
    test_scores_basemodel = calculate_nodewise_error_scores(test_predictions,
                                                  test_ground_truth,
                                                  smoothing_window,
                                                  epsilon)
    
    test_scores_basemodel = np.max(test_scores_basemodel, axis=0)
    test_scores_proposed = test_attention_score.sum(axis=1)
    
    for a in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        test_scores = a*test_scores_basemodel + test_scores_proposed*(1-a)
        precision_point, recall_point, f1_point  = test_perf_based_on_best(test_scores, test_anomaly_label)
        precision_range, recall_range, f1_range = best_threshold_rangewise_pa(
                                                    test_scores, test_anomaly_label,
                                                    threshold_steps=1000,
                                                    max_quantile=0.99,
                                                    greater_is_anomaly=True)

        roc_auc, prc_auc = test_roc_prc_perf(test_scores, test_anomaly_label)
        
        print("attention ratio: {} Precision_point: {:.4f} Recall_point: {:.4f} F1_point: {:.4f} Precision_range: {:.4f} Recall_range: {:.4f} F1_range: {:.4f} ROC: {:.4f}".
            format(a, precision_point, recall_point, f1_point, precision_range, recall_range, f1_range, roc_auc))

    
    
def test_roc_prc_perf(test_scores, anomaly_labels):
    # test_scores: [num_nodes, total_time_len]
    # anomaly_labels: [total_time_len]
    fpr, tpr, _ = roc_curve(anomaly_labels, test_scores)
    roc_auc = auc(fpr, tpr)
    prc_auc = average_precision_score(anomaly_labels, test_scores)
    return roc_auc, prc_auc


def test_perf_based_on_best(test_scores, anomaly_labels, threshold_steps= 1000) -> tuple:
    # find the best threshold based on the f1 score of the test set
    min_score = np.min(test_scores)
    max_score = np.quantile(test_scores, 0.99)
    best_f1 = 0
    best_threshold = 0
    
    for step in range(threshold_steps):
        threshold = min_score + (max_score - min_score) * step / threshold_steps
        predicted_labels = (test_scores > threshold).astype(int)
        f1 = f1_score(anomaly_labels, predicted_labels)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    final_predicted_labels = (test_scores > best_threshold).astype(int)
    precision_point = precision_score(anomaly_labels, final_predicted_labels)
    recall_point = recall_score(anomaly_labels, final_predicted_labels)
    f1_point = f1_score(anomaly_labels, final_predicted_labels)
    
    return precision_point, recall_point, f1_point


def point_adjust_pa(pred, gt):
    # expands point-wise predictions to range-wise predictions by marking an entire anomaly segment as detected 
    # if at least one point within the segment is correctly detected.

    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True

            for j in range(i, -1, -1):
                if gt[j] == 0:
                    break
                pred[j] = 1

            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                pred[j] = 1

        elif gt[i] == 0:
            anomaly_state = False

        if anomaly_state:
            pred[i] = 1

    return pred


def best_threshold_rangewise_pa(
    scores,
    gt,
    threshold_steps=1000,
    max_quantile=0.99,
    greater_is_anomaly=True
):
    # sweeps thresholds over anomaly scores, applies point-adjusted evaluation, 
    # and selects the threshold that maximizes range-wise F1 on the given data.
    scores = np.asarray(scores)
    gt = np.asarray(gt).astype(int)

    min_score = float(np.min(scores))
    max_score = float(np.quantile(scores, max_quantile))

    best_f1 = -1.0
    best_thr = None
    best_pred_point = None
    best_pred_range = None
    best_cm = None
    best_p = best_r = None

    for step in range(threshold_steps + 1):
        thr = min_score + (max_score - min_score) * step / threshold_steps

        if greater_is_anomaly:
            pred_point = (scores > thr).astype(int)
        else:
            pred_point = (scores < thr).astype(int)

        pred_range = point_adjust_pa(pred_point, gt)

        f1 = f1_score(gt, pred_range, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
            best_pred_point = pred_point
            best_pred_range = pred_range
            best_p = precision_score(gt, pred_range, zero_division=0)
            best_r = recall_score(gt, pred_range, zero_division=0)
            best_cm = confusion_matrix(gt, pred_range)

    return  best_p, best_r, best_f1