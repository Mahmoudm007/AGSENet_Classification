# AGSENet: Road Ponding Classification Adaptation

This repository implements a **multi-class image classification** model adapted from the original semantic segmentation architecture described in the paper *"AGSENet: A Robust Road Ponding Detection Method for Proactive Traffic Safety"*.

While the original paper focused on dense pixel-level delineation of water patches, this implementation captures the architectural essence of AGSENet—specifically its powerful saliency extraction and multi-scale feature enhancement—and explicitly repurposes it for **5-class road surface condition recognition**.

## Key Architectural Adaptations

### What Was Preserved
1. **Multi-Stage RSU Encoder**: Re-implemented the U2Net-inspired blocks (RSU-7, RSU-6, RSU-5, RSU-4, RSU-4F) maintaining intra-block multi-scale aggregation.
2. **CSIF Module**: The **Channel Saliency Information Focus** module is preserved directly after each encoder stage. It contains both the SCIP (spatial contextual perception via depthwise approximations) and CSII (attention-based channel interactions).
3. **SSIE Module**: The **Spatial Saliency Information Exploration** module still acts as an intelligent refinement stage. It applies noise rejection (via subtraction) and edge refinement (Hadamard product), weighting features with spatial attention maps.

### What Was Changed for Classification
1. **Classification Head replacing Deep Supervision Decoder**: Dense upsampling decoders and intermediate BCE+Dice loss functions were completely removed.
2. **Top-Down Fusion Neck**: Instead of an expansive decoder, the SSIE fusion modules act as a lightweight, top-down feature refining neck. Adjacent scales are fused (stage 6 -> stage 5 -> ...).
3. **Global Multi-Scale Output**: The outputs of deeper scales (f3, f4, f5, f6) undergo Adaptive Average *and* Max Pooling, are concatenated, and map through a lightweight MLP.
4. **Imbalance-Aware CrossEntropy Loss**: The pipeline was switched from dense maps to a robust sequence of image-label metrics (`WeightedCrossEntropy` and `Multiclass FocalLoss`).
5. **Description-Aligned Auxiliary Learning**: The AGSENet classifier now supports a frozen CLIP text branch that encodes the full class descriptions, aligns image embeddings with those class-description prototypes, and mixes text-similarity logits with the main visual classifier.

## Expected Directory Structure

``` text
g:/GeoTrans/Ponding/AGSENet_Classification/
├── Data/
│   └── 1 Defined/
│       ├── 0 Bare/
│       ├── 1 Centre - Partly/
│       ├── 2 Two Track - Partly/
│       ├── 3 One Track - Partly/
│       └── 4 Fully/
├── configs/
│   └── default.yaml
├── models/
├── data/
├── utils/
├── train.py
├── evaluate.py
├── infer.py
└── README.md
```

## Setup
Install the necessary dependencies via:
```bash
pip install -r requirements.txt
```

## Imbalance-Aware Loss Functions
Because road conditions span widely varying frequencies, this project dynamically computes sample weights per class during dataloading and can also rebalance batches with a weighted sampler. In `configs/default.yaml`, control these settings:

- `loss_name`: Choose `"weighted_cross_entropy"` (default) or `"class_balanced_focal"`.
- `class_weight_mode`: Choose between `"inverse"`, `"median_frequency"`, `"normalized_inverse"`, `"balanced"`, or `"effective_num"`.
- `use_weighted_sampler`: Enable a `WeightedRandomSampler` so under-represented classes appear more evenly across training batches.
Weights are calculated **only** from the training split.

## Full Dataset Explainability (Grad-CAM)
When evaluating the model, you can export Grad-CAM heatmaps for the **entire** validation or test dataset to visually inspect whether the model detects road tracks or off-road artifacts. 
- Overlays are saved dynamically in `outputs/gradcam/{split}/{class}/`.

## Usage Instructions

### Training
```bash
python train.py --config configs/default.yaml
```
Output models and CSV logs will be placed in the newly created `outputs/` directory.

### Evaluation & Explainability
Evaluate the validation set and generate heatmaps for every image using the correct class labels:
```bash
python evaluate.py --checkpoint outputs/checkpoints/best_model.pth --split val --export-gradcam --gradcam-target true
```
The `--export-gradcam` flag delegates to `visualize.py` and exports encoder/decoder Grad-CAM overlays into `outputs/gradcam/{split}/{class}/Encoder 1..N`, `Decoder 1..N`, `Combined_Encoders`, and `Combined_Decoder`.

### Ad-hoc Inference
Run prediction on a single image or an entire folder of raw images:
```bash
python infer.py --checkpoint outputs/checkpoints/best_model.pth --input path/to/image.jpg
python infer.py --checkpoint outputs/checkpoints/best_model.pth --input path/to/folder/ --output_csv predictions.csv
```

### Advanced Visualization Pipeline (Paper-style Figures)
We provide a dedicated `visualize.py` script to generate comprehensive, publication-style visual outputs as seen in the original AGSENet paper. This includes:
1. **Validation/Test Image Panels**: Side-by-side original image with GT and Predicted labels, colored (green/red) based on correctness.
2. **Encoder Stage Heatmaps**: A standalone figure visualizing the CNN feature activations out of every individual encoder stage (Encoder 1 through 6).
3. **Decoder/Fusion Stage Heatmaps**: A standalone figure visualizing the top-down feature fusion process through the SSIE modules.
4. **ROC, Precision-Recall, Calibration, and Top-K Panels**: Multiclass ROC/AUC, precision-recall curves, confidence calibration, normalized confusion, and per-image top-k prediction panels.

```bash
# Export all available components on the validation set
python visualize.py --split val \
    --export-image-panels \
    --export-encoder-gradcam \
    --export-decoder-gradcam \
    --export-roc \
    --export-pr \
    --export-topk \
    --export-calibration \
    --best-checkpoint auto
```
*Note: Because this project tackles classification rather than semantic segmentation, Decoder heatmaps represent the multi-scale classification fusion path rather than expanding dense map outputs.* Grad-CAM overlays are saved under `outputs/gradcam/{split}/`, original image exports are saved under `outputs/actual_images/{split}/`, and summary plots plus top-k panels remain under `outputs/visualizations/{split}/`.
