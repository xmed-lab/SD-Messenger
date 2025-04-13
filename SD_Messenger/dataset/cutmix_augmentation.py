import numpy as np
import torch


def mask_to_bounding_box(mask):
    """
    Convert a semantic mask (PyTorch tensor) into a bounding box that covers all foreground areas.

    Args:
        mask (torch.Tensor): Input semantic mask tensor. Shape: (H, W)

    Returns:
        torch.Tensor: Bounding box coordinates [xmin, ymin, xmax, ymax]. Shape: (4,)
    """
    # Find the indices of foreground pixels
    indices = torch.nonzero(mask)

    if len(indices) == 0:
        # No foreground pixels found, return an empty bounding box
        return torch.zeros(4)

    # Extract the coordinates of foreground pixels
    xmin = indices[:, 1].min()
    ymin = indices[:, 0].min()
    xmax = indices[:, 1].max()
    ymax = indices[:, 0].max()

    # Create the bounding box tensor
    bounding_box = torch.tensor([xmin, ymin, xmax, ymax])

    return bounding_box


def cutmix_segmentation(images_a, masks_a, images_b, masks_b, cut_mix_size):
    # print(f'img shape: {images_a.shape}, mask shape: {masks_a.shape}')
    foreground_area = mask_to_bounding_box(masks_b)
    # print(f'foreground area: {foreground_area}')
    if foreground_area[2] - foreground_area[0] > cut_mix_size and foreground_area[3] - foreground_area[1] > cut_mix_size:
        height, width = images_a.shape[1], images_a.shape[2]

        cut_h = np.int64(cut_mix_size)
        cut_w = np.int64(cut_mix_size)

        cx = np.random.randint(low=foreground_area[0], high=foreground_area[2])
        cy = np.random.randint(low=foreground_area[1], high=foreground_area[3])
        bbx1 = np.clip(cx - cut_w // 2, 0, width)
        bby1 = np.clip(cy - cut_h // 2, 0, height)
        bbx2 = np.clip(cx + cut_w // 2, 0, width)
        bby2 = np.clip(cy + cut_h // 2, 0, height)
        # print(f'cx: {cx}, cut_w: {cut_w}, bbx1: {bbx1}')
        images_a[:, bby1:bby2, bbx1:bbx2] = images_b[:, bby1:bby2, bbx1:bbx2]
        masks_a[bby1:bby2, bbx1:bbx2] = masks_b[bby1:bby2, bbx1:bbx2]

        # lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (height * width))
        # print(f'cutted {torch.tensor([bbx1, bby1, bbx2, bby2])}')
        return images_a, masks_a, torch.tensor([bbx1, bby1, bbx2, bby2])
    else:
        return images_a, masks_a, torch.tensor([0, 0, 0, 0])
