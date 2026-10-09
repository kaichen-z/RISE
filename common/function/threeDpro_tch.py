import torch

def cluster2point(threed_clusters, threed_labels, threed_mags, center):
    sel_mag, sel_mean, sel_range, sel_theta, sel_shape, sel_num, sel_lab = [],[],[],[],[],[],[]
    for label in torch.unique(threed_labels):
        cluster = threed_clusters[threed_labels == label]
        mags = threed_mags[threed_labels == label]
        sel_mag.append(torch.max(mags))
        sel_mean.append(torch.mean(cluster.float(), dim=0))
        sel_range.append(torch.norm(sel_mean[-1] - center))
        sel_theta.append(compute_angle(sel_mean[-1] - center, torch.zeros(2) - center))
        width = torch.max(cluster[:, 0]) - torch.min(cluster[:, 0])
        length = torch.max(cluster[:, 1]) - torch.min(cluster[:, 1])
        sel_shape.append(length / width if width != 0 else torch.tensor(0))
        sel_num.append(len(cluster))
        sel_lab.append(label)
    return torch.Tensor(sel_mag), torch.stack(sel_mean), torch.Tensor(sel_range), \
            torch.Tensor(sel_theta), torch.Tensor(sel_shape), torch.Tensor(sel_num), torch.Tensor(sel_lab)
    
def compute_angle(u, v):
    dot_product = torch.dot(u, v)
    magnitude_u = torch.norm(u)
    magnitude_v = torch.norm(v)
    cos_theta = dot_product / (magnitude_u * magnitude_v)
    angle_radians = torch.acos(torch.clamp(cos_theta, -1.0, 1.0))
    return angle_radians
    
def checkSelfOverlap(sel_mean, sel_mag, distance_threshold=10):
    point_position, point_mag = sel_mean, sel_mag  # Shape: (N, 2)
    # Step 1: Compute pairwise distances
    dists = torch.cdist(point_position, point_position)  # Pairwise distances, shape: (N, N)
    # Step 2: Get indices of close points
    close_mask = (dists < distance_threshold) & (dists > 0)  # Exclude self-distances
    # Step 3: Compare magnitudes for close points
    keep_mask = torch.ones(len(point_position), dtype=torch.bool)  # Start by keeping all points
    for i in range(len(point_position)):
        if not keep_mask[i]:  # Skip already removed points
            continue
        close_indices = torch.where(close_mask[i])[0]
        for j in close_indices:
            if point_mag[j] > point_mag[i]:
                keep_mask[i] = False
                break
        keep_mask[close_indices[point_mag[close_indices] <= point_mag[i]]] = False
    return keep_mask
    
def deleteClose(select_range, select_theta, real_range, real_theta, simi_theta=0.075 * torch.pi):
    mask1 = select_range > real_range
    mask2 = torch.abs(select_theta - real_theta) > simi_theta
    if len(select_range[mask1*mask2]) > 0: mask3 = torch.argmin(select_range[mask1*mask2])
    else: mask3 = True
    index_list = torch.arange(len(select_range))
    fianl_mask = index_list[mask1*mask2][mask3]
    mask = torch.zeros(len(select_range))
    mask[fianl_mask] = 1
    return mask.bool()

if __name__ == "__main__":
    point_position = torch.tensor([[0, 0], [1, 1], [9, 9], [10, 10], [11,11]], dtype=torch.float32)  # Shape: (N, 2)
    point_mag = torch.tensor([10, 20, 5, 15, 20], dtype=torch.float32) 
    mask = checkSelfOverlap(point_position, point_mag)
    print(mask)