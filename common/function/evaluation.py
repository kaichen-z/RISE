import numpy as np
from scipy.spatial import cKDTree

def chamfer_distance_2d(point_cloud1_PRED, point_cloud2_GT, unit=50):
    measure = 10**(-2)
    point_cloud1_PRED, point_cloud2_GT = point_cloud1_PRED/unit, point_cloud2_GT/unit
    tree1 = cKDTree(point_cloud1_PRED)
    tree2 = cKDTree(point_cloud2_GT)
    # Compute forward distances (P -> Q)
    forward_dists, _ = tree1.query(point_cloud2_GT)
    forward_sum = np.sum(forward_dists**2)
    # Compute backward distances (Q -> P)
    backward_dists, _ = tree2.query(point_cloud1_PRED)
    backward_sum = np.sum(backward_dists**2)
    # Chamfer Distance
    chamfer_dist = forward_sum + backward_sum
    return chamfer_dist*measure

def f1_score_2d(point_cloud1_PRED, point_cloud2_GT, unit=50, threshold=0.05):
    measure = 100
    point_cloud1_PRED, point_cloud2_GT = point_cloud1_PRED/unit, point_cloud2_GT/unit
    # Build KD-Trees for fast nearest-neighbor lookup
    tree1 = cKDTree(point_cloud1_PRED)
    tree2 = cKDTree(point_cloud2_GT)
    # Match points in point_cloud1 to point_cloud2
    distances1, _ = tree1.query(point_cloud2_GT)
    tp1 = np.sum(distances1 <= threshold)  # True Positives from point_cloud2's perspective
    # Match points in point_cloud2 to point_cloud1
    distances2, _ = tree2.query(point_cloud1_PRED)
    tp2 = np.sum(distances2 <= threshold)  # True Positives from point_cloud1's perspective
    # Precision and Recall
    precision = tp2 / len(point_cloud1_PRED) if len(point_cloud1_PRED) > 0 else 0
    recall = tp1 / len(point_cloud2_GT) if len(point_cloud2_GT) > 0 else 0
    # F1 Score
    if precision + recall == 0:
        return 0
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score*measure

def compute_iou_dice(points1, points2, shape):
    mask1 = np.zeros(shape, dtype=np.uint8)
    mask2 = np.zeros(shape, dtype=np.uint8)
    # Fill the masks based on points
    for x, y in points1:
        mask1[y, x] = 1  # OpenCV/NumPy uses (row=y, col=x)
    for x, y in points2:
        mask2[y, x] = 1
    # Compute intersection and union
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    # Compute IoU and Dice
    iou = intersection / union if union != 0 else 0
    dice = (2 * intersection) / (mask1.sum() + mask2.sum()) if (mask1.sum() + mask2.sum()) != 0 else 0
    return iou, dice

def filter_negative(points, H=600):
    condition = (points[:, 0] >= 0) & (points[:, 0] < H) & (points[:, 1] >= 0) & (points[:, 1] < H)
    return points[condition], condition

import torch

def evaluate_obs(prediction, ground_truth):
    # Computing IOU and DICE for obstacle computation.
    prediction = (prediction > 0).float()
    ground_truth = (ground_truth > 0).float()
    prediction = prediction.view(-1)
    ground_truth = ground_truth.view(-1)
    intersection = torch.sum(prediction * ground_truth) * 1.8
    union = torch.sum(prediction) + torch.sum(ground_truth) - intersection
    iou = intersection / union if union > 0 else 0.0
    dice = (2 * intersection) / (torch.sum(prediction) + torch.sum(ground_truth)) if (torch.sum(prediction) + torch.sum(ground_truth)) > 0 else 0.0
    return iou.item()*100, dice.item()*100

def evaluate_wall(prediction, ground_truth, threshold=5):
    # Computing chamfer distance and F1 score for wall computation.
    prediction = (prediction > 0).float()
    ground_truth = (ground_truth > 0).float()
    pred_points = torch.nonzero(prediction)  # Shape (N, 2)
    gt_points = torch.nonzero(ground_truth)  # Shape (M, 2)
    if pred_points.shape[0] == 0 or gt_points.shape[0] == 0:
        chamfer, f1_score = float('inf'), float('inf')
    '''--------- chamfer distance ---------'''
    dists_pred_to_gt = torch.cdist(pred_points.float(), gt_points.float(), p=2)  # (N, M)
    dists_gt_to_pred = torch.cdist(gt_points.float(), pred_points.float(), p=2)  # (M, N)
    min_pred_to_gt = torch.min(dists_pred_to_gt, dim=1)[0]  # (N,)
    min_gt_to_pred = torch.min(dists_gt_to_pred, dim=1)[0]  # (M,)
    chamfer = torch.mean(min_pred_to_gt) + torch.mean(min_gt_to_pred)
    '''--------- f1 score ---------'''
    dists_pred_to_gt = torch.cdist(pred_points.float(), gt_points.float(), p=2)  # (N, M)
    dists_gt_to_pred = torch.cdist(gt_points.float(), pred_points.float(), p=2)  # (M, N)
    tp = (torch.min(dists_pred_to_gt, dim=1)[0] < threshold).float().sum()
    fp = pred_points.shape[0] - tp
    fn = gt_points.shape[0] - (torch.min(dists_gt_to_pred, dim=1)[0] < threshold).float().sum()
    precision = tp / (tp + fp + 1e-6)  # Avoid division by zero
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    return chamfer.item(), f1.item()*100

def evaluate_wall2(prediction, ground_truth, threshold=5):
    # Computing chamfer distance and F1 score for wall computation.
    prediction = (prediction > 0).float()
    ground_truth = (ground_truth > 0).float()
    pred_points = torch.nonzero(prediction)  # Shape (N, 2)
    gt_points = torch.nonzero(ground_truth)  # Shape (M, 2)
    '''--------- mean error ---------'''
    if pred_points.shape[0] == 0 or gt_points.shape[0] == 0:
        return float('inf')  # No valid comparison
    dists_pred_to_gt = torch.cdist(pred_points.float(), gt_points.float(), p=2)  # (N, M)
    dists_gt_to_pred = dists_pred_to_gt.T  # (M, N)
    mean_error = torch.mean(torch.min(dists_pred_to_gt, dim=1)[0]) + torch.mean(torch.min(dists_gt_to_pred, dim=1)[0])
    '''--------- 90 ---------'''
    if pred_points.shape[0] == 0 or gt_points.shape[0] == 0:
        return float('inf')  # No valid comparison
    dists_pred_to_gt = torch.cdist(pred_points.float(), gt_points.float(), p=2)  # (N, M)
    dists_gt_to_pred = dists_pred_to_gt.T  # (M, N)
    min_pred_to_gt = torch.min(dists_pred_to_gt, dim=1)[0]  # (N,)
    min_gt_to_pred = torch.min(dists_gt_to_pred, dim=1)[0]  # (M,)
    all_errors = torch.cat([min_pred_to_gt, min_gt_to_pred])  # Combine both directions
    percentile_90_error = torch.quantile(all_errors, 0.9)  # Compute 90th percentile
    return mean_error.item(), percentile_90_error.item()

if __name__ == "__main__":
    # Example usage
    point_cloud1_PRED = np.random.rand(100, 2)  # Replace with your 2D point cloud
    point_cloud2_GT = np.random.rand(120, 2)  # Replace with your 2D point cloud
    cd = chamfer_distance_2d(point_cloud1_PRED, point_cloud2_GT)
    print(f"Chamfer Distance: {cd}")