# Laboratory-Work-5-Activity-Comparative-Analysis

A. Model Performance
1. Which pre-trained model achieved the highest accuracy? Why?
MobileNetV2 achieved the highest accuracy at 95.00% (validation). This is because MobileNetV2 uses depthwise separable convolutions and inverted residual blocks, making it highly efficient at extracting features while maintaining a lightweight architecture. Its design optimizes for both accuracy and speed, allowing it to learn discriminative features effectively even with transfer learning (frozen base layers).
2. Which model had the lowest performance? What could be the reason?
ResNet50 had the lowest performance at 17.10% validation accuracy. This poor performance could be due to several factors: (1) ResNet50's skip connections may not align well with the custom head layers when base layers are frozen; (2) The model may have suffered from vanishing gradients in the unfrozen configuration; (3) ResNet50 requires careful input preprocessing (ImageNet normalization) which may not have been applied correctly; (4) The model's complexity (25.6M parameters) may need more fine-tuning to adapt to the grain dataset.
3. How did loss values compare across models?
MobileNetV2 had the lowest loss (train: 0.3710, val: 0.2312), indicating confident and accurate predictions. VGG16 had moderate loss (train: 1.6072, val: 1.4574). ResNet50 had the highest loss (train: 2.8225, val: 2.8035), barely improving from random initialization. Lower loss correlates with higher accuracy across all models.
B. Evaluation Metrics
4. Why is accuracy not enough to evaluate a model?
Accuracy alone can be misleading, especially with imbalanced datasets. A model could achieve decent accuracy by predicting majority classes while failing on minority classes. Precision, recall, and F1-score provide per-class performance, while AUC measures the model's ability to rank predictions across all thresholds. For example, ResNet50 had 17% accuracy but an AUC of 0.52 (barely better than random 0.50), revealing it truly learned nothing.
5. Which model had the best F1-score? What does it indicate?
MobileNetV2 had the best F1-score with macro average 0.75 and weighted average 0.76. This indicates an excellent balance between precision (minimizing false positives) and recall (minimizing false negatives). Most classes had F1-scores above 0.75, with millet_pearl at 0.91, showing strong and consistent performance across all grain types.
6. How did Precision and Recall differ across models?
MobileNetV2 showed balanced precision (0.76 macro avg) and recall (0.76 macro avg), meaning it correctly identified most true positives without excessive false alarms. VGG16 had lower precision (0.37) than recall (0.32), indicating more false positives. ResNet50 had catastrophic precision (0.06) and recall (0.13), with many classes showing 0.00 precision — meaning it failed to predict those classes entirely.
C. Confusion Matrix Analysis
7. Which classes were frequently misclassified?
From the confusion matrices:
VGG16: Significant confusion between visually similar grains — barley vs. buckwheat, emmer vs. einkorn, and quinoa vs. amaranth. Maize_corn and wheat had poor recall.
ResNet50: Nearly random predictions — most predictions concentrated in a few classes (barley, buckwheat, millet_finger) while ignoring others completely.
MobileNetV2: Minimal misclassification. Some confusion between rye (F1: 0.34) and wheat (F1: 0.61), and triticale (F1: 0.64) with other wheat varieties, likely due to visual similarity.
8. What patterns did you observe in the confusion matrix?
MobileNetV2 shows a strong diagonal pattern with dark blue concentrated on correct predictions, indicating accurate classification. VGG16 shows a blurred diagonal with significant off-diagonal values, indicating moderate confusion between similar classes. ResNet50 shows no clear pattern — predictions are scattered randomly, confirming the model failed to learn meaningful features.
D. ROC and AUC
9. Which model had the highest AUC score?
MobileNetV2 had the highest AUC at 0.99 (micro-average). Individual class AUCs were exceptional — amaranth (1.00), buckwheat (0.99), barley (0.96). VGG16 had moderate AUC at 0.89, while ResNet50 had poor AUC at 0.52.
10. What does AUC tell us about model performance?
AUC measures the probability that a randomly chosen positive instance is ranked higher than a randomly chosen negative instance. MobileNetV2's AUC of 0.99 means it has a 99% chance of correctly ranking a positive grain sample above a negative one — near-perfect discrimination. ResNet50's 0.52 is essentially random guessing (50% chance), confirming complete failure to learn.
E. Explainability (Grad-CAM)
11. What did Grad-CAM reveal about model decision-making?
Grad-CAM revealed that:
MobileNetV2 focused intensely on the grain heads/seed clusters (the red/yellow hotspots), particularly the flowering/seed-bearing parts of the plant.
VGG16 showed broader, more diffuse activation across the plant, including some background regions.
ResNet50 had scattered, unfocused activation that didn't clearly align with any specific plant feature.
12. Did the model focus on relevant image regions?
Yes for MobileNetV2 — it focused on the grain heads and seed structures, which are the most discriminative features for grain classification. The heatmap clearly highlights the purple seed clusters and flowering parts. VGG16 partially focused on relevant regions but with more background interference. ResNet50 showed no meaningful focus.
13. Which model produced the most meaningful heatmaps?
MobileNetV2 produced the most meaningful heatmaps. The activation was tightly concentrated on the grain-producing parts of the plant (the purple amaranth seed heads), with minimal background activation. This aligns perfectly with its superior classification performance and demonstrates that it learned biologically relevant features.
F. Model Comparison & Improvement
14. Which model would you recommend for deployment? Why?
MobileNetV2 is the clear choice for deployment because:
Highest accuracy (95%) with excellent generalization (val loss < train loss)
Lightweight architecture — designed for mobile and edge devices
Fast inference — depthwise separable convolutions reduce computation
Strong AUC (0.99) — reliable ranking across all classes
Focused Grad-CAM — proves it learns meaningful, trustworthy features
15. How can you further improve your best-performing model?
Improvements could include:
Fine-tuning — unfreeze top layers of MobileNetV2 and train with lower learning rate (1e-5)
More data — collect additional images for underperforming classes (rye, wheat, triticale)
Data augmentation — add rotation, zoom, brightness adjustments to increase diversity
Ensemble methods — combine MobileNetV2 with EfficientNetB0 for boosted performance
Class balancing — use weighted loss for classes with lower F1-scores
Longer training — increase epochs with early stopping patience
G. Real-World Application
16. How can your model be applied in real-world scenarios?
This grain classification model can be applied in:
Agricultural quality control — automated sorting of grain shipments by species
Food safety inspection — detecting contamination or mislabeling in grain products
Seed certification — verifying seed purity for farmers and distributors
Research — rapid identification of grain varieties in botanical studies
Supply chain transparency — tracking grain origin and type from farm to consumer
17. What are the risks of deploying an inaccurate model?
Risks include:
Economic losses — misclassified grains sold at wrong prices
Food safety issues — contaminated or allergenic grains entering supply chain
Legal liability — false certification of seed purity
Erosion of trust — users lose confidence in AI-assisted decisions
ResNet50-level failure — a model with 17% accuracy would be worse than human inspection
18. How can this system be integrated into a mobile/web app?
Integration steps:
Convert to TensorFlow Lite — optimize MobileNetV2 for mobile (quantization, pruning)
Build mobile app — use TensorFlow Lite Interpreter on Android/iOS
Web deployment — use TensorFlow.js for browser-based inference
API backend — deploy model on cloud (AWS/GCP) with REST API for image upload
Real-time camera — integrate with device camera for instant grain identification
Offline capability — MobileNetV2's small size (~14MB) enables offline use

----------------------------------------
| Model       | Train Acc | Train Loss | Test Acc | Test Loss | ROC AUC |
| :---------- | :-------- | :--------- | :------- | :-------- | :------ |
| VGG16       | 57.18%    | 1.6072     | 75.60%   | 1.4574    | 0.89    |
| ResNet50    | 13.32%    | 2.8225     | 17.10%   | 2.8035    | 0.52    |
| MobileNetV2 | 90.18%    | 0.3710     | 95.00%   | 0.2312    | 0.99    |

-----------------------------------------
https://colab.research.google.com/drive/1GQojdDSotugDAvxO_zIBTCSfq_FpbLCr#scrollTo=qZa4ErG15Dzw
