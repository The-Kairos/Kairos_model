"""
Adapted from Moment-DETR standalone_eval/utils.py
Original: https://github.com/jayleicn/moment_detr/blob/main/standalone_eval/utils.py
Originally from MMAction2: https://github.com/open-mmlab/mmaction2
"""
import json
import numpy as np
from sklearn.metrics import precision_recall_curve


def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]


def compute_temporal_iou_batch_paired(pred_windows, gt_windows):
    intersection = np.maximum(
        0, np.minimum(pred_windows[:, 1], gt_windows[:, 1]) - np.maximum(pred_windows[:, 0], gt_windows[:, 0])
    )
    union = np.maximum(pred_windows[:, 1], gt_windows[:, 1]) \
            - np.minimum(pred_windows[:, 0], gt_windows[:, 0])
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union != 0)


def compute_temporal_iou_batch_cross(spans1, spans2):
    areas1 = spans1[:, 1] - spans1[:, 0]
    areas2 = spans2[:, 1] - spans2[:, 0]

    left = np.maximum(spans1[:, None, 0], spans2[None, :, 0])
    right = np.minimum(spans1[:, None, 1], spans2[None, :, 1])

    inter = np.clip(right - left, 0, None)
    union = areas1[:, None] + areas2[None, :] - inter

    iou = inter / union
    return iou, union


def interpolated_precision_recall(precision, recall):
    mprecision = np.hstack([[0], precision, [0]])
    mrecall = np.hstack([[0], recall, [1]])
    for i in range(len(mprecision) - 1)[::-1]:
        mprecision[i] = max(mprecision[i], mprecision[i + 1])
    idx = np.where(mrecall[1::] != mrecall[0:-1])[0] + 1
    ap = np.sum((mrecall[idx] - mrecall[idx - 1]) * mprecision[idx])
    return ap


def compute_average_precision_detection(ground_truth,
                                        prediction,
                                        tiou_thresholds=np.linspace(
                                            0.5, 0.95, 10)):
    num_thresholds = len(tiou_thresholds)
    num_gts = len(ground_truth)
    num_preds = len(prediction)
    ap = np.zeros(num_thresholds)
    if len(prediction) == 0:
        return ap

    num_positive = float(num_gts)
    lock_gt = np.ones((num_thresholds, num_gts)) * -1
    prediction.sort(key=lambda x: -x['score'])
    tp = np.zeros((num_thresholds, num_preds))
    fp = np.zeros((num_thresholds, num_preds))

    ground_truth_by_videoid = {}
    for i, item in enumerate(ground_truth):
        item['index'] = i
        ground_truth_by_videoid.setdefault(item['video-id'], []).append(item)

    for idx, pred in enumerate(prediction):
        if pred['video-id'] in ground_truth_by_videoid:
            gts = ground_truth_by_videoid[pred['video-id']]
        else:
            fp[:, idx] = 1
            continue

        _pred = np.array([[pred['t-start'], pred['t-end']], ])
        _gt = np.array([[gt['t-start'], gt['t-end']] for gt in gts])
        tiou_arr = compute_temporal_iou_batch_cross(_pred, _gt)[0]

        tiou_arr = tiou_arr.reshape(-1)
        tiou_sorted_idx = tiou_arr.argsort()[::-1]
        for t_idx, tiou_threshold in enumerate(tiou_thresholds):
            for j_idx in tiou_sorted_idx:
                if tiou_arr[j_idx] < tiou_threshold:
                    fp[t_idx, idx] = 1
                    break
                if lock_gt[t_idx, gts[j_idx]['index']] >= 0:
                    continue
                tp[t_idx, idx] = 1
                lock_gt[t_idx, gts[j_idx]['index']] = idx
                break

            if fp[t_idx, idx] == 0 and tp[t_idx, idx] == 0:
                fp[t_idx, idx] = 1

    tp_cumsum = np.cumsum(tp, axis=1).astype(float)
    fp_cumsum = np.cumsum(fp, axis=1).astype(float)
    recall_cumsum = tp_cumsum / num_positive

    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)

    for t_idx in range(len(tiou_thresholds)):
        ap[t_idx] = interpolated_precision_recall(precision_cumsum[t_idx, :],
                                                  recall_cumsum[t_idx, :])
    return ap


def get_ap(y_true, y_predict, interpolate=True, point_11=False):
    assert len(y_true) == len(y_predict), "Prediction and ground truth need to be of the same length"
    if len(set(y_true)) == 1:
        if y_true[0] == 0:
            return 0
        else:
            return 1
    else:
        assert sorted(set(y_true)) == [0, 1], "Ground truth can only contain elements {0,1}"

    precision, recall, _ = precision_recall_curve(y_true, y_predict)
    recall = recall.astype(np.float32)

    if interpolate:
        for i in range(1, len(precision)):
            precision[i] = max(precision[i - 1], precision[i])

    if point_11:
        precision_11 = [precision[np.where(recall >= t)[0][-1]] for t in np.arange(0, 1.01, 0.1)]
        return np.mean(precision_11)
    else:
        indices = np.where(np.diff(recall))
        return np.mean(precision[indices])
