# -identifying-and-classifying-rocks-
 identifying and classifying rocks by determining  their mineral composition

Here is two model
1.mineral classification using CNN of multiclassification 
click this link to download dataset of multiclassification dataset from kaggle 
https://www.kaggle.com/datasets/asiedubrempong/minerals-identification-dataset

2.image detection model yolov8.s using image detection dataset from roboflow yolov8.s 
click here to download data from robloflow
https://universe.roboflow.com/finalproject-flqsu/mineral_ore_detection/dataset/2/download



about the project 
 identifying and classifying rocks by determining
 their mineral composition. However, traditional manual methods involving microscopic
 observations are time-consuming, labor-intensive, and prone to human error. As
 industries such as mining, construction, and research require fast and large-scale
 geological analysis, the need for automated tools has become more pressing to improve
 efficiency and precision.

📌 Project Overview

This project implements an AI-powered Mineral Detection System using the YOLOv8m (You Only Look Once v8 medium) object detection model. The system can detect and classify multiple mineral types from images. The trained model is deployed using Streamlit for real-time image classification.

Supported Minerals:

🟤 Baryte (BaSO₄) → Used in drilling fluids.

⚪ Calcite (CaCO₃) → Forms limestone & marble.

🟣 Fluorite (CaF₂) → Famous for fluorescence.

🟡 Pyrite (FeS₂) → Known as Fool’s Gold.

📂 Dataset

Custom dataset prepared for 4-class mineral classification.

Each class contains images + bounding box annotations.

Dataset split:

70% Training

20% Validation

10% Testing

⚙️ Model Training

We trained the model using YOLOv8m, which balances accuracy and speed.

🔹 Key Parameters

Epochs: 50 (longer training for convergence)

Batch Size: 32 (stable gradient updates)

Image Size: 640 (balanced between speed & accuracy)

Optimizer: AdamW (better generalization than SGD)

Learning Rate: 0.001 with lrf=0.1 decay

Augmentations: Mixup, Mosaic, HSV shifts, rotation, scaling

📊 Model Evaluation

Evaluation performed using YOLOv8’s built-in validation.

Metrics:

Precision, Recall, mAP@50, mAP@50-95

Confusion Matrix (class-wise detection performance)

Training Curves (loss, precision, recall, mAP)

Loss curves (train/box_loss, train/cls_loss, train/dfl_loss, and their validation counterparts) are decreasing smoothly, which means the model learned well without severe overfitting.

Precision improves steadily and stabilizes around 0.75–0.78.

Recall improves gradually, stabilizing around 0.70–0.75.

mAP@50 reaches ~0.80, and mAP@50-95 around 0.63–0.65, which is decent performance.
 
 
