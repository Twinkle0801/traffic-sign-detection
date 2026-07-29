def compute_iou(box1, box2):
    """boxes as (x1, y1, x2, y2) in pixel coordinates"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

# Example: ground-truth box vs a predicted box that's slightly off
ground_truth = (100, 100, 200, 200)   # a perfect 100x100 box
prediction_good = (105, 105, 205, 205)  # shifted slightly — high IoU
prediction_bad = (150, 150, 250, 250)   # shifted a lot — low IoU

print(f"Good prediction IoU: {compute_iou(ground_truth, prediction_good):.3f}")
print(f"Bad prediction IoU:  {compute_iou(ground_truth, prediction_bad):.3f}")
print()
print("mAP50  = average precision using IoU >= 0.50 as the 'correct' threshold")
print("mAP50-95 = average precision averaged across IoU thresholds 0.50 to 0.95")
print("         (a stricter, more comprehensive measure of box tightness)")