import torch

def obs_metrics_torch(prediction, ground_truth):
    # Computing IOU and DICE for obstacle computation.
    prediction = (prediction > 0).float()
    ground_truth = (ground_truth > 0).float()
    prediction = prediction.view(-1)
    ground_truth = ground_truth.view(-1)
    intersection = torch.sum(prediction * ground_truth)
    union = torch.sum(prediction) + torch.sum(ground_truth) - intersection
    iou = intersection / union if union > 0 else 0.0
    dice = (2 * intersection) / (torch.sum(prediction) + torch.sum(ground_truth)) if (torch.sum(prediction) + torch.sum(ground_truth)) > 0 else 0.0
    return iou, dice

def wall_metrics_torch(prediction, ground_truth, threshold=1.25):
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
    return chamfer.item(), f1.item()