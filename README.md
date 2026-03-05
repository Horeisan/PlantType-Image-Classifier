# **Plant-Type-Image-Classifier**

## **Project Description**
This project demonstrates the training and benchmarking of image classification models using **PyTorch** and **PyTorch Lightning**. The analysis is performed on the Plant Type Dataset to classify plant images into their correct categories.

The main focus of the study is to compare a **SimpleCNN** baseline with a **pretrained VGG16 model fine-tuned** for plant classification and to evaluate how pretrained features improve generalization. Key components include:

- **Data Preparation:** Loading the dataset with **PyTorch Dataset/DataLoader** and applying resizing, normalization, and training-time augmentation (flip + rotation) to improve robustness.

- **Model Benchmarking:** Training a **SimpleCNN** from scratch and comparing it against a **pretrained VGG16** model with a replaced classification head to match the plant classes.

- **Transfer Learning Setup:** Using **ImageNet pretrained weights** and training the classifier layers (with the option to **unfreeze later layers** for extra fine-tuning).

- **Training Control (Lightning):** Using a **LightningModule** with `training_step`, `validation_step`, and `test_step`, plus **ModelCheckpoint** and **EarlyStopping** to keep the best model and reduce overfitting.

- **Evaluation and Error Analysis:** Reporting test performance with **loss**, **accuracy**, **precision**, **recall**, **F1-score**, and a **confusion matrix**, and inspecting difficult test samples to understand common mistakes.
