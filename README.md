# Transfer Learning with CNNs (Image Classification)

## 📌 Project Overview

Training deep Convolutional Neural Networks (CNNs) from scratch requires massive datasets, significant computational power, and long training times. This project overcomes these limitations by utilizing **Transfer Learning**.

By leveraging a pre-trained CNN (MobileNetV2) that has already learned to extract complex visual features from millions of images in the ImageNet dataset, we can achieve high accuracy on a completely new image classification task with minimal training time and a smaller dataset.

## 📊 Dataset

* **Dataset:** TensorFlow Flowers Dataset
* **Classes:** 5 (Daisy, Dandelion, Roses, Sunflowers, Tulips)
* **Data Split:** The dataset is split into 80% training data and 20% validation data.
* **Image Size:** All images are resized to `224x224` pixels to match the expected input shape of the MobileNetV2 architecture.

## ⚙️ Methodology

1. **Data Preparation & Loading:** Images are loaded efficiently using `image_dataset_from_directory`. Performance is optimized using `tf.data.AUTOTUNE` for prefetching.
2. **Base Model Loading (Transfer Learning):** `MobileNetV2` is loaded without its top classification layer (`include_top=False`) and initialized with pre-trained `imagenet` weights.
3. **Feature Extraction:** The pre-trained base model is frozen (`trainable = False`), meaning its weights will not be updated during initial training.
4. **Custom Classification Head:** A custom head is added consisting of:
* A `GlobalAveragePooling2D` layer to flatten the extracted features.
* A `Dropout(0.2)` layer to mitigate overfitting.
* A `Dense` output layer with softmax activation for 5-class prediction.


5. **Model Training:** The model is compiled using the Adam optimizer and trained using sparse categorical cross-entropy loss.

## 📈 Evaluation Metrics

The model's performance on the validation set is analyzed using:

* **Overall Accuracy & Loss:** Tracked across epochs to monitor learning and check for overfitting.
* **Classification Report:** Providing Precision, Recall, and F1-Score for each specific flower class.
* **Confusion Matrix:** Visualizing exactly where the model is making incorrect predictions between similar classes.




## 🛠️ Technologies Used

* **Language:** Python
* **Deep Learning Framework:** TensorFlow / Keras
* **Libraries:** NumPy, Scikit-Learn, Matplotlib, Seaborn
* **Pre-trained Architecture:** MobileNetV2

