# Model Evaluation Summary

**Model:** YOLOv8n, trained 30 epochs, 416x416, CPU
**Dataset:** 15-class traffic sign dataset (4,969 images), Roboflow "Self-Driving Cars" v4

## Validation Set Results
- mAP50: 0.937 | mAP50-95: 0.809 | Precision: 0.939 | Recall: 0.882

## Test Set Results (held out, never used in training)
- mAP50: 0.910 | mAP50-95: 0.780 | Precision: 0.882 | Recall: 0.849

The small, consistent drop from validation to test (roughly 3-6 points across all four
metrics) indicates the model generalizes well rather than overfitting to the validation split.

## Strongest classes
Stop (0.995 AP50) and most Speed Limit signs (0.86-0.98 AP50 on test) — consistently strong
across both validation and test splits.

## Weakest classes
- **Red Light** — 0.779 AP50 (val) -> 0.690 AP50 (test). Consistently the weakest class across
  both splits, confirming this is a real model limitation, not a fluke of one data split.
- **Green Light** — 0.777 AP50 (val) -> 0.816 AP50 (test). Also weak, though slightly more
  stable than Red Light.
- Root cause (confirmed via confusion matrix): both light classes generate a large number of
  false positives against background regions (1580 for Green Light, 1169 for Red Light on the
  validation confusion matrix) — far higher than any other class. Traffic lights vary more in
  scale, angle, and lighting condition across a scene than flat, standardized road signs do,
  making them inherently harder to distinguish from background clutter.

## Known limitations
- **Speed Limit 10** has minimal data: only 19 training images, 0 validation instances, and
  just 3 test instances. On the 3 available test instances it scored 0.806 AP50 but only 0.528
  recall (missed about half), though this estimate is too noisy (n=3) to draw strong
  conclusions from. This class would benefit most from additional training data if the model
  were to be improved further.
- Model was trained and evaluated entirely on CPU; larger models (yolov8s/m) or longer training
  were not explored due to hardware constraints.

## Real-World / Out-of-Distribution Testing
Tested on 14 genuinely novel images gathered from general web image search (not from the
training/validation/test dataset in any way) — a deliberate mix of stop signs, speed limit
signs, traffic lights, multi-sign scenes, an image from Frankfurt, Germany, and two heavily
watermarked stock photos.

- Correctly detected with high confidence: 12/14 images (all Stop signs 0.94-0.97, all Speed
  Limit signs 0.97, most Red Light detections 0.72-0.89)
- Weak/low-confidence detections: 2 instances (Green Light 0.29, one Red Light 0.26) — both
  isolated to the two classes already flagged as weakest during validation/test evaluation
- No false positives observed; no genuinely missed signs
- Notably robust to: heavy stock-photo watermarking, multi-sign scenes (2 signs on one pole),
  a real photo from a country (Germany) not necessarily represented in training data

**Conclusion:** the model generalizes well beyond its own dataset for its 13 strong classes.
Its two weakest classes (Green Light, Red Light) remained weak on entirely novel images too —
consistent with the confusion-matrix diagnosis (high background false-positive rate for both
light classes) rather than a fluke of any one data split. This is a well-understood, documented
limitation rather than an unpredictable failure mode.

## Video & Live Webcam Testing
- Video inference on an 8-second real clip (a Stop sign filmed with heavy sun glare behind it):
  detected correctly on most frames (0.84 confidence even under harsh backlight), with
  intermittent gaps during frames where glare was most intense — an explainable, realistic
  hard case rather than random flickering.
- Live webcam test with a single real traffic light held up to the camera: correctly detected
  Red Light at 0.97 confidence.
- Live webcam test with a printed multi-sign sheet: correctly detected Stop (0.96) and
  Red Light (0.55, small/angled icon), correctly produced no detection for a Parking sign
  (not a trained class - correct behavior). One misclassification: a "Speed Limit 50" sign
  was labeled "Speed Limit 40" at only 0.30 confidence - close to the 0.25 threshold,
  suggesting the model was genuinely uncertain rather than confidently wrong. Likely due to
  visual similarity between "40" and "50" at small/angled scale rather than a systemic
  confusion between these classes on the test-set numbers (both scored 0.964-0.970 AP50
  individually).

## Discovered Limitation: Vehicle-Category Speed Signs

During live webcam testing, a genuinely new failure mode was found: **vehicle-category speed
limit signs** (e.g. Indian highway signs combining a vehicle icon such as a truck or car with
the speed number) were confidently misclassified — a "60" sign was labeled "Speed Limit 50" at
0.93 confidence. This is a high-confidence wrong answer, not just low-confidence uncertainty,
indicating a genuine gap in training data coverage rather than an ambiguous edge case. Root
cause: the training dataset only contains plain European-style circular signs (number only,
no vehicle icon), so the model has never seen this visual format.

### Fine-tuning attempt
A small experiment was run to test whether targeted fine-tuning could close this gap:
- Gathered and manually labeled 4 images containing 7 vehicle-category speed signs, mapped to
  existing classes based on their actual printed number (vehicle icon ignored)
- Fine-tuned the existing `best.pt` for 10 epochs at a low learning rate (0.0005) specifically
  to avoid catastrophic forgetting of existing performance
- **Result on main test set: no degradation** (mAP50 0.910 -> 0.912, effectively unchanged) —
  confirms the conservative learning rate successfully protected existing performance
- **Result on the new sign type: inconclusive** (1 of 3 signs correctly reclassified in a
  follow-up test image; one wrong, one missed entirely) — not a reliable improvement

**Conclusion:** 4 images was not enough data for the model to robustly learn this new visual
pattern. The experiment was safe (no harm to existing capability) but not sufficient to close
the gap. A proper fix would require gathering and labeling 30-50+ varied examples ofa
vehicle-category signs across different numbers and icon types before attempting another
fine-tune. This remains a documented, understood limitation rather than a silently broken
capability. The fine-tuned checkpoint was not promoted to the main model (`models/best.pt`)
given the inconclusive result; it exists only as an experimental artifact.