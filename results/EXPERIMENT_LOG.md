# SCRIPT Research Experiment Log

| Experiment ID |Model / Configuration | Feature Space | Evaluation Protocol | Overall ROC-AUC | Overall PR-AUC | Blur AUC | Dropout AUC | Thickening AUC | Thinning AUC |
|:---|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **EXP-00-BASELINE-DK** | Devi & Kavya (2023) [DCT + ResNet-18] | 2D DCT (128-D) + ResNet-18 (512-D) | One-Class vs N=1000 synth | **0.5656** | **0.4330** | 0.5963 | 0.5162 | 0.5693 | 0.5822 |
| **EXP-00-BASELINE-AG** | Agarwal et al. (2023) [Faithful Reimplementation] | 8-D Handcrafted Geometric | One-Class vs N=1000 synth | **0.5819** | **0.4772** | 0.6265 | 0.5250 | 0.5950 | 0.5858 |
| **EXP-00-BASELINE-AL** | Alevizos et al. (2024) [Adapted Autoencoder] | 6-D Geometric Handwriting Features | Conv-LSTM AE (MSE loss) | **0.6066** | **0.4998** | 0.5934 | 0.6007 | 0.6305 | 0.6005 |
| **EXP-00-BASELINE-KNN**| Proposed Method (Ablation: k-NN) | 256-D Fused Multimodal Embedding | Euclidean k-NN (k=5) | **0.6241** | **0.4851** | 0.6788 | 0.5462 | 0.6285 | 0.6468 |
| **EXP-00-BASELINE-LOF**| **Proposed Method (Upgraded: Adaptive LOF)** | **256-D Fused Multimodal Embedding** | **Adaptive LOF** | **0.6552** | **0.5669** | **0.7321** | **0.5743** | **0.6728** | **0.6508** |
| **EXP-02-ABL-A** | Ablation A: Clinical/OpenCV Only | 9-D Handcrafted Geometric Features | Adaptive LOF | **0.5998** | **0.4853** | 0.6416 | 0.5489 | 0.5761 | 0.6341 |
| **EXP-02-ABL-B** | Ablation B: ResNet-50 Visual Only | 2048-D Deep Visual Features | Adaptive LOF | **0.5920** | **0.4662** | 0.6499 | 0.5282 | 0.5840 | 0.6108 |
| **EXP-02-ABL-C** | Ablation C: Raw Concatenation | 2057-D Visual + Geometric | Adaptive LOF | **0.5920** | **0.4664** | 0.6500 | 0.5284 | 0.5837 | 0.6109 |
| **EXP-02-ABL-D** | Ablation D: Early Fusion (No Bottleneck) | 2112-D Visual + Encoded CV | Adaptive LOF | **0.5933** | **0.4696** | 0.6475 | 0.5318 | 0.5847 | 0.6134 |
| **EXP-02-ABL-E** | **Ablation E: Proposed Bottleneck** | **256-D Multimodal Bottleneck** | **Adaptive LOF** | **0.6552** | **0.5669** | **0.7321** | **0.5743** | **0.6728** | **0.6508** |
| **EXP-02-ABL-F** | Ablation F: Fused + k-NN | 256-D Multimodal Bottleneck | Euclidean k-NN (k=5) | **0.6241** | **0.4851** | 0.6788 | 0.5462 | 0.6285 | 0.6468 |
| **EXP-02-ABL-G** | Ablation G: Fused + OC-SVM | 256-D Multimodal Bottleneck | One-Class SVM (RBF, nu=0.05) | **0.6110** | **0.4937** | 0.6486 | 0.5567 | 0.6220 | 0.6200 |
| **EXP-02-ABL-H** | Ablation H: Fused + Isolation Forest | 256-D Multimodal Bottleneck | Isolation Forest (T=200) | **0.5867** | **0.4449** | 0.6317 | 0.5349 | 0.5792 | 0.6046 |
