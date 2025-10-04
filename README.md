**🖼️ ViT-CIFAR10 (Hybrid CNN + Vision Transformer)**
📌 Project Overview

This project implements a Hybrid Vision Transformer (ViT) for CIFAR-10 classification.
We combine a CNN backbone for patch embedding with a Transformer encoder and additional CLS Token refinement for improved global representations.

📊 Results

Model Variant	Test Accuracy (%)

Baseline ViT 	71.5

Hybrid ViT 	85.36

**🔑 Concise Analysis**

Patch size (4×4 after CNN downsampling): Smaller patches → richer detail but higher compute. We chose 4×4 as a balance between granularity and efficiency on CIFAR-10.

Depth/Width trade-off: 6 layers, 8 heads, 256-dim embedding = good balance of accuracy vs. stability.

CLS Token Refinement: Extra attention on [CLS] token improved classification by ~+2%.

Augmentation: Random crop, flip, jitter, rotation → prevents overfitting and boosts robustness.

Optimizer/Scheduler: AdamW + CosineAnnealingLR ensures smooth convergence.

Overlapping vs. Non-overlapping patches: CNN-based embedding provides overlap and stable features, better than raw patch splitting.

**Main Changes That Improved Accuracy**

CNN-based Patch Extraction
Instead of directly cutting 32×32 patches, we use a CNN:

32×32×3 → Conv → 32×32×64  
         → Conv → 32×32×128  
         → MaxPool → 16×16×128  
         → Conv (4×4, stride=4) → 4×4×256  
         → Flatten → 16 patches × 256 embedding  


→ This overlapping CNN approach captures both local textures and global context, which works better for smaller datasets like CIFAR-10.

CLS Token Refinement
The [CLS] token aggregates information from all patch tokens. Passing it through extra ClassAttention + MLP blocks improved accuracy by about +2%, since classification relies on richer global information.

Augmentation Strategy
Final augmentations (crop, flip, jitter, rotation) were tuned experimentally, resulting in stronger generalization performance.
***********************************************************************************************************************************************************************************************************
**Segment Anything Model v2 (SAM2) Pipeline**

🧠 Detailed Pipeline Description

This project implements a Grounded SAM2 system that combines Grounding DINO for text-conditioned detection and SAM 2 (Segment Anything Model 2) for segmentation.
The pipeline allows segmenting specific objects in an image based on natural language prompts.

**1. Input Processing**

The user provides:

An image (as a NumPy array or PIL image)

A text prompt (e.g., "car", "tyre", "helmet")

The text is tokenized and encoded using Grounding DINO’s text encoder:

text_encoding = wrapper.encode_tokenized_text(wrapper.tokenize(["tyre"]))


The image is preprocessed and converted into tensors matching model requirements.

**2. Object Detection (Grounding DINO)**

Grounding DINO performs phrase grounding — detecting objects that match the given text prompt.

It outputs:

Bounding boxes ([cx, cy, w, h] normalized)

Logits (confidence scores)

Corresponding text phrases

These bounding boxes are then transformed to absolute pixel coordinates:

boxes = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])


This ensures the detections align with the original image dimensions.

**3. Region-based Segmentation (SAM2)**

The processed bounding boxes are passed as input prompts to the SAM2 predictor.

SAM2 then generates fine-grained segmentation masks for each detected region:

mask, scores, _ = self.sam_model.predict(
    point_coords=None,
    point_labels=None,
    box=boxes.numpy(),
    multimask_output=False,
)


For each box, the output includes:

The binary segmentation mask

Confidence scores

Object label inferred from the text query

**4. Output Aggregation**

The pipeline returns a structured output per image:

{
  "masks": [tensor],
  "boxes": [tensor],
  "logits": [tensor],
  "labels": [string]
}


These can be used for visualization, evaluation, or downstream computer vision tasks.

**5. Optional: Automatic Mask Generation**

When no text prompt is given, SAM2AutomaticMaskGenerator can be used to produce region proposals automatically across the image, functioning as a fully unsupervised segmentation stage.

⚙️ Integrated Components
Component	Role
Grounding DINO	Text-conditioned object detection (box-level)
SAM 2	High-quality segmentation from boxes or points
GroundedSAM2Wrapper	Custom interface that unifies both models for end-to-end inference
📉 Limitations

Latency: Sequential model execution (DINO → SAM2) causes slower inference for large batches.

Dependency on Detection Accuracy: Poor detections from Grounding DINO directly degrade SAM2 segmentation quality.

GPU Memory Usage: High-resolution images may exceed VRAM limits during mask prediction.

Prompt Sensitivity: Text queries must be clear and unambiguous; vague prompts yield incorrect detections.

No Real-time Inference Yet: Current pipeline runs offline; real-time streaming support is planned.

Limited Multi-object Handling: Works best for 1–5 objects; dense scenes reduce segmentation precision.
