# Final Model Prediction and OOD Detection Report

## 1. Executive Summary

This report documents the final evaluation notebook, `FinalPredict_OOD.ipynb`, for the project's 8-class vehicle image classification model.

The notebook uses a pretrained ResNet18 whose final classification layer contains eight known vehicle classes:

- `ambulance`
- `autobus`
- `kamyun`
- `kamyunet`
- `minibus`
- `savari`
- `taxi`
- `vanet`

The main purpose of the final notebook is not only to measure ordinary 8-class classification performance, but also to evaluate whether an unseen vehicle category, **Neysan/Nissan**, can be rejected as out-of-distribution (OOD).

Two complementary OOD scores are used:

1. **Energy score**
2. **Mahalanobis distance** in the 512-dimensional ResNet18 feature space

The notebook combines these scores with model confidence to assign every image one of four evaluation statuses:

- `known`
- `needs_review`
- `unknown`
- `known_failure`

The fourth status is used only when a sample is known to be a true Neysan/Nissan image but the OOD detector accepts it as a known class.

A particularly important design decision is that the standalone Neysan dataset is evaluated **outside** the normal 8-class `ImageFolder`. This preserves the original 8-class model definition and prevents Neysan from accidentally becoming a ninth model class.

---

## 2. Model Definition

The final notebook loads the checkpoint:

`checkpoints/resnet18_full_finetuning_non_letterbox_best.pt`

The architecture is:

- ImageNet-pretrained ResNet18
- Final fully connected layer replaced by an 8-output linear layer
- Checkpoint weights loaded from the final full-finetuning experiment
- Model evaluated in inference mode

The model therefore produces probabilities only for the eight known classes. There is no explicit `Neysan` output neuron.

This is important for OOD evaluation: Neysan is not expected to be classified into a dedicated ninth class. Instead, the OOD detector must identify Neysan as an input that does not sufficiently resemble the known training distribution.

---

## 3. Input Preprocessing

The notebook uses the same preprocessing pipeline for test-time evaluation as the final model:

1. Letterbox the image to `224 × 224`
2. Convert the image to a tensor
3. Apply ImageNet normalization

The normalization parameters are:

- Mean: `[0.485, 0.456, 0.406]`
- Standard deviation: `[0.229, 0.224, 0.225]`

Using the same preprocessing during evaluation is essential because the trained ResNet18 expects the same input representation used during model development.

---

## 4. Dataset Protocol

The notebook supports several test configurations.

| Dataset | Intended structure | Evaluation |
|---|---|---|
| `test0` | Eight known classes | Standard classification evaluation |
| `test1_2` | Eight known classes + explicit Neysan | 8-class evaluation plus explicit Neysan OOD evaluation |
| `test2` | Eight folders with Neysan mixed into known folders | OOD suspicion; exact Neysan metrics require ground truth |
| `test3` | Eight folders with Neysan mixed across classes | OOD suspicion; exact Neysan metrics require ground truth |
| `test4` | Mixed/relabeled data including Neysan | OOD suspicion; exact semantic metrics require reliable ground truth |
| `dataset/neysan` | Standalone Neysan images | Dedicated standalone OOD benchmark |

The notebook deliberately keeps `dataset/neysan` separate from the normal 8-class test dataset.

This is the correct design for an 8-class classifier. Adding a `neysan` folder directly to the normal eight-class dataset would cause `ImageFolder` to create a ninth class and would no longer represent the intended 8-class test protocol.

---

## 5. Ground-Truth Limitations for Mixed Datasets

For datasets such as `test2`, `test3`, and `test4`, the folder name is not necessarily the semantic truth.

For example, if a true Neysan image is physically stored inside the `vanet` folder, `ImageFolder` will naturally assign the folder label `vanet`.

Therefore:

- The notebook can calculate the folder-label classification result.
- The notebook can calculate Energy and Mahalanobis OOD scores.
- The notebook can flag suspicious images.
- But it cannot prove that a particular suspicious image is Neysan without an external ground-truth annotation.

The notebook therefore supports an optional `GROUND_TRUTH_CSV`.

The required column is:

`image_path, is_true_nissan`

An optional `true_class` column can also provide the corrected semantic class.

This distinction is critical when reporting Neysan detection performance.

---

## 6. Feature Extraction for OOD Detection

The notebook extracts the 512-dimensional feature vector immediately before the ResNet18 classification layer.

The feature extraction path is:

`conv1 → bn1 → relu → maxpool → layer1 → layer2 → layer3 → layer4 → avgpool → flatten`

The resulting feature vector has 512 dimensions.

These features are used for Mahalanobis-distance-based OOD detection.

---

## 7. Mahalanobis Feature Bank

The feature bank is constructed only from the known training dataset.

For each of the eight known classes, the notebook calculates a class mean in the 512-dimensional feature space.

Thus, the model has eight class centroids:

- one centroid for `ambulance`
- one for `autobus`
- one for `kamyun`
- one for `kamyunet`
- one for `minibus`
- one for `savari`
- one for `taxi`
- one for `vanet`

The global covariance matrix is estimated using **Ledoit-Wolf shrinkage**.

The inverse covariance, or precision matrix, is then used to calculate Mahalanobis distances.

For each image, the notebook calculates the distance to every known-class centroid and uses the minimum distance as the final Mahalanobis score.

A large minimum distance indicates that the image is far from all known-class feature distributions.

---

## 8. Energy-Based OOD Score

The notebook uses the energy function:

`E(x) = -T logsumexp(logits / T)`

with:

`T = 1.0`

The score is calculated directly from the eight model logits.

The Energy score is used as a complementary signal to the feature-space Mahalanobis distance.

Using two different OOD signals is useful because the two methods measure different aspects of uncertainty:

- Energy examines the model's output logits.
- Mahalanobis examines the learned feature representation.

---

## 9. OOD Threshold Calibration

The OOD thresholds are calibrated using **known validation data only**.

This is an important methodological choice.

The notebook does not use Neysan images to select the thresholds.

For both Energy and Mahalanobis scores, the threshold is the 95th percentile of the known validation distribution:

`threshold = quantile(validation_scores, 0.95)`

The configured quantile is:

`QUANTILE = 0.95`

Therefore, Neysan is not used to tune the detector.

Once the thresholds are calculated, they are fixed and applied to:

- the selected test dataset
- the standalone Neysan benchmark

This makes the standalone Neysan evaluation a more meaningful OOD test.

---

## 10. Final Decision Rule

The notebook uses the following decision hierarchy.

### Case 1: Both OOD detectors flag the image

If:

- Energy says OOD, and
- Mahalanobis says OOD

then:

`status = unknown`

This is the strongest OOD decision.

### Case 2: Only one detector flags the image

If exactly one detector says OOD:

`status = needs_review`

The sample is suspicious but does not have enough evidence for the strongest `unknown` decision.

### Case 3: Neither detector flags OOD, but confidence is low

If both OOD detectors accept the sample but:

`confidence < 0.70`

then:

`status = needs_review`

This prevents low-confidence predictions from automatically being treated as reliable known-class predictions.

### Case 4: Neither detector flags OOD and confidence is sufficiently high

If:

`confidence >= 0.70`

and neither OOD detector flags the sample:

`status = known`

The model considers the image a normal known-class prediction.

---

## 11. Meaning of `known_failure`

For a normal known-class image, `known` means the image passed the OOD rules.

For a verified true Neysan image, however, the interpretation changes.

The notebook contains:

```python
if status == "known":
    status = "known_failure"
```

during standalone Neysan evaluation.

Therefore:

> **`known_failure` means a true Neysan image was accepted by the OOD system as if it belonged to the known 8-class distribution.**

This is the most important OOD failure case.

For example, if a true Neysan image receives:

- Energy OOD = False
- Mahalanobis OOD = False
- confidence >= 0.70

then it first receives `known`, but because its true class is Neysan, it is converted to:

`known_failure`

This means the OOD detector missed the unseen class.

---

## 12. Important Issue in the Current Final Summary

The current final-summary cell contains:

```python
"Neysan Known Failures": int(
    (results_df["status"] == "known_failure").sum()
),
```

This line does **not** correctly summarize the standalone Neysan benchmark.

`results_df` is the result table for the selected `test_loader`.

The standalone Neysan benchmark is stored separately in:

`standalone_nissan_results`

Therefore, if Neysan is kept in `dataset/neysan` and is not explicitly included in the selected test dataset, `results_df` contains no true Neysan samples.

In that situation:

`(results_df["status"] == "known_failure").sum()`

can correctly evaluate to zero **for that particular DataFrame**, but the label `Neysan Known Failures` is misleading.

It does **not** mean that the OOD detector had zero Neysan failures.

It means only that the selected test DataFrame contained zero rows with the status `known_failure`.

### Correct interpretation

The standalone Neysan failure count should instead be calculated from:

`standalone_nissan_results`

For example:

```python
"Neysan Known Failures": int(
    (standalone_nissan_results["status"] == "known_failure").sum()
)
```

with an empty-data check if necessary.

The corresponding failure rate should be:

`Neysan Known Failures / Total Standalone Neysan Images`

This is the correct metric for the standalone Neysan benchmark.

### Conclusion about the current `0.000000`

If the final summary shows:

`Neysan Known Failures = 0.000000`

while the standalone Neysan evaluation contains true Neysan images, the value is **not a correct Neysan failure metric**.

It is a consequence of the summary cell using the wrong DataFrame.

The standalone Neysan section itself is correctly designed to produce `known_failure` values. The error is specifically in the final aggregation/reporting cell.

---

## 13. Standalone Neysan Evaluation

The standalone benchmark is implemented using a dedicated dataset loader.

It supports both:

```text
dataset/neysan/*.jpg
```

and:

```text
dataset/neysan/neysan/*.jpg
```

The images are passed through the same trained 8-class model.

For each image, the notebook records:

- image path
- true class = `neysan`
- predicted known class
- confidence
- Energy score
- Mahalanobis score
- Energy OOD flag
- Mahalanobis OOD flag
- final status
- explicit Neysan ground-truth indicator
- source dataset

The model itself remains an eight-class model throughout this process.

---

## 14. Neysan Detection Categories

For the standalone benchmark, the output categories have a direct interpretation.

### `unknown`

Both OOD detectors rejected the image.

This is the strongest successful Neysan/OOD detection category.

### `needs_review`

Only one OOD detector rejected the image, or the confidence was below 70%.

This is a partially successful detection that requires human review.

### `known_failure`

Neither OOD detector rejected the image and the model confidence was at least 70%.

The image is truly Neysan, but the system accepted it as a known vehicle.

This is an OOD miss.

---

## 15. Known-Class Performance

For ordinary 8-class evaluation, the notebook calculates:

- Mean Cross-Entropy Loss
- Classification Accuracy
- Wrong predictions
- Confusion matrix
- Normalized confusion matrix
- Per-class precision
- Per-class recall
- Per-class F1-score

Only samples with reliable known-class ground truth are used for the standard classification metrics.

Explicit Neysan samples are excluded from the normal 8-class loss and accuracy calculations.

This is necessary because Neysan is not one of the eight model output classes.

---

## 16. Wrong Prediction Analysis

The notebook creates:

`wrong_predictions.csv`

This file contains known-class samples for which:

`true_class != predicted_class`

The table includes:

- image path
- true class
- predicted class
- confidence
- Energy
- Mahalanobis distance
- final status

This allows every known-class classification error to be inspected individually.

---

## 17. Grad-CAM Analysis

The notebook implements Grad-CAM using:

`model.layer4`

The Grad-CAM visualization is used to understand which image regions influenced the model's prediction.

Grad-CAM outputs are generated for:

1. Wrong known predictions
2. Unknown images
3. Needs-review images
4. Neysan images detected as unknown
5. Neysan images detected as needs review
6. Neysan images missed by the OOD detector

The notebook displays only a limited number of images for readability while saving individual outputs for the complete groups.

This is useful because the OOD results can be investigated visually rather than being treated as only numerical scores.

---

## 18. Unknown Images

The `unknown` subset contains images for which both Energy and Mahalanobis detectors report OOD.

The notebook saves these results to:

`unknown_predictions.csv`

The associated Grad-CAM outputs are saved under the unknown-image Grad-CAM directory.

For mixed datasets where true Neysan labels are unavailable, these images should be described as **OOD-suspected**, not automatically as confirmed Neysan.

---

## 19. Needs-Review Images

The `needs_review` subset contains images that require additional inspection.

A sample enters this group when:

- exactly one OOD detector flags it, or
- neither detector flags it but confidence is below 0.70

The notebook saves:

`needs_review_predictions.csv`

This group is particularly useful for practical deployment because it represents cases where an automated decision is less trustworthy.

---

## 20. Neysan Grad-CAM Analysis

The notebook separates Neysan samples into:

- Neysan detected as `unknown`
- Neysan marked `needs_review`
- Neysan missed as `known_failure`

The third group is especially important.

The `neysan_missed` Grad-CAM results show what the model focuses on when it incorrectly treats an unseen Neysan image as a known vehicle.

This can help identify why the learned representation does not sufficiently separate Neysan from the known vehicle classes.

---

## 21. Confusion Matrices

Two confusion matrices are generated for known-class evaluation:

1. Raw-count confusion matrix
2. Row-normalized confusion matrix

The raw matrix shows the number of images assigned to every predicted class.

The normalized matrix shows the fraction of each true class assigned to each predicted class.

These matrices are only meaningful for reliable known-class ground truth.

They should not be interpreted as a true semantic confusion matrix for mixed datasets when Neysan images are hidden inside known folders and no external ground-truth manifest is available.

---

## 22. Classification Report

The notebook generates a standard sklearn classification report containing:

- precision
- recall
- F1-score
- support

for all eight known classes.

The report is saved both as CSV and text output.

This provides a more detailed view of per-class performance than accuracy alone.

---

## 23. Saved Outputs

The notebook uses:

`results/FinalPredict_OOD/`

with subdirectories for:

- CSV results
- Grad-CAM outputs
- saved images

Important CSV outputs include:

- `FinalPredict_OOD_results.csv`
- `wrong_predictions.csv`
- `unknown_predictions.csv`
- `needs_review_predictions.csv`
- `ood_suspected_images.csv`
- `standalone_neysan_results.csv`
- `confusion_matrix_known.csv`
- `confusion_matrix_normalized.csv`
- `classification_report.csv`
- `final_summary.csv`

The exact contents depend on the selected dataset and whether external ground truth is provided.

---

## 24. Reproducibility

The notebook uses:

- the final ResNet18 checkpoint
- the project's merged known training data
- a fixed train/validation split with seed `7`
- a fixed OOD quantile of `0.95`
- a fixed confidence threshold of `0.70`
- deterministic test-loader ordering (`shuffle=False`)

The OOD thresholds are calculated from known validation data rather than tuned on Neysan.

This makes the evaluation protocol consistent and reproducible.

---

## 25. Interpretation of the Final System

The final system should be understood as a **two-stage decision mechanism**.

### Stage 1: Known-class classification

The ResNet18 predicts one of the eight known classes.

### Stage 2: OOD screening

Energy and Mahalanobis scores determine whether the image appears sufficiently similar to the known training distribution.

The final status therefore does not simply equal the softmax prediction.

A model can have very high softmax confidence and still be considered OOD if both OOD detectors reject the sample.

Conversely, a true Neysan image can receive a high-confidence known-class prediction if both OOD detectors fail to reject it. Such an image becomes a `known_failure`.

This distinction is central to the project's OOD objective.

---

## 26. Limitations

Several limitations must be considered when interpreting the results.

### 26.1 Neysan is not a learned class

The model has no Neysan output neuron.

Therefore, Neysan detection is entirely dependent on the OOD mechanism.

### 26.2 OOD thresholds are percentile-based

The 95th percentile threshold is a practical calibration choice, not a universally optimal threshold.

Different quantiles can produce different trade-offs between false OOD detections and missed OOD samples.

### 26.3 Mixed datasets need semantic ground truth

If Neysan images are placed inside known-class folders, folder labels cannot establish the true semantic class.

An external annotation manifest is required for exact Neysan detection metrics.

### 26.4 High softmax confidence is not proof of correctness

A neural classifier can assign extremely high confidence to an OOD image.

This is precisely why Energy and Mahalanobis screening are included.

### 26.5 The final summary currently contains an aggregation bug

The `Neysan Known Failures` metric in the final summary currently reads from `results_df` instead of `standalone_nissan_results`.

This should be corrected before treating `final_summary.csv` as the authoritative report of Neysan OOD performance.

---

## 27. Recommended Final Summary Metrics

The final report should distinguish the following quantities.

### Known-class evaluation

- Known test sample count
- Known test loss
- Known test accuracy
- Number of wrong known predictions
- Per-class precision/recall/F1
- Confusion matrix

### OOD evaluation

- Energy threshold
- Mahalanobis threshold
- Number of `unknown`
- Number of `needs_review`
- Number of `known_failure`
- OOD/review detection rate on verified Neysan
- OOD miss rate on verified Neysan

For standalone Neysan, the recommended formulas are:

`Neysan Detection Rate = (unknown + needs_review) / total Neysan`

and:

`Neysan OOD Miss Rate = known_failure / total Neysan`

These two numbers together provide a much clearer picture of OOD performance than a single count.

---

## 28. Final Assessment

The overall notebook design is appropriate for the intended 8-class vehicle classification problem with an unseen Neysan category.

The strongest parts of the design are:

1. The model remains an 8-class classifier.
2. Neysan is evaluated independently rather than incorrectly added to the known-class training/test mapping.
3. OOD thresholds are calibrated only on known validation data.
4. Energy and Mahalanobis detection provide complementary signals.
5. A review category is included instead of forcing every uncertain image into `known` or `unknown`.
6. True Neysan images that pass the OOD detector are explicitly marked as `known_failure`.
7. Per-image CSV results and Grad-CAM visualizations make the system auditable.
8. Mixed datasets are explicitly documented as requiring external ground truth for exact semantic Neysan metrics.

The main issue that must be fixed before the notebook is considered fully final is the `Neysan Known Failures` field in `final_summary.csv`.

The current expression:

```python
(results_df["status"] == "known_failure").sum()
```

measures known failures in the selected test DataFrame, not in the standalone Neysan benchmark.

For a standalone Neysan evaluation, the authoritative source is:

```python
standalone_nissan_results
```

Therefore, the current `0.000000` value should **not** be interpreted as zero Neysan OOD failures.

After correcting that aggregation, the notebook's final summary will accurately reflect both the standard 8-class classification performance and the standalone Neysan OOD performance.

---

## 29. Final Conclusion

The final evaluation framework provides a complete pipeline for:

**8-class classification → OOD scoring → uncertainty triage → error analysis → Grad-CAM interpretation → CSV export**

The most important conceptual distinction is between:

- a normal known-class error,
- an OOD image correctly rejected,
- an uncertain image requiring review,
- and a true Neysan image that the OOD system incorrectly accepts as known.

The last category, `known_failure`, is the critical failure mode for the unseen-class problem.

Accordingly, all final Neysan performance claims should be based on `standalone_nissan_results` when Neysan is evaluated as the independent `dataset/neysan` benchmark, and not on the generic `results_df` used for the selected 8-class test dataset.
