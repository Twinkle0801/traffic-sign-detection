# Model Evaluation Summary

**Model:** YOLOv8n, trained 30 epochs, 416x416, CPU
**Dataset:** 15-class traffic sign dataset (4,969 images), Roboflow "Self-Driving Cars" v4

## Validation Set Results
- mAP50: 0.937 | mAP50-95: 0.809 | Precision: 0.939 | Recall: 0.882

## Test Set Results (held out, never used in training)
- mAP50: [fill in] | mAP50-95: [fill in] | Precision: [fill in] | Recall: [fill in]

## Strongest classes
Speed Limit signs (10 excluded — no eval data) and Stop sign: 0.93–1.00 AP50

## Weakest classes
- Green Light (0.777 AP50) — high false-positive rate against background
- Red Light (0.779 AP50) — same pattern
- Root cause: traffic lights vary more in scale/lighting/angle than flat standardized signs, confirmed via confusion matrix (1580 and 1169 background false-positives respectively)

## Known limitations
- "Speed Limit 10" has minimal training data (19 images) and zero validation instances — untested class, likely underperforms in practice