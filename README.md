# Project Title: Image Segmentation and Deep Learning Model Training

## Introduction

Hello, In this project, I have explored and implemented several deep learning models for image segmentation, including U-Net, Enhanced U-Net, FCN (Fully Convolutional Network), and Attention U-Net. My goal is to achieve better performance on semantic segmentation tasks using these models and to provide a detailed analysis and visualization of their performance.

## Dataset

I used a custom dataset comprising BMP format raw images and PNG format labels. The dataset paths are as follows:
- Raw images: `/content/drive/MyDrive/image_10`
- Labels: `/content/drive/MyDrive/image_10_label`

The data underwent standard preprocessing steps during import, including resizing, normalization, and binarization of labels.

## Model Architectures

In this project, I have built and trained the following deep learning models:

### 1. **U-Net**

U-Net is a widely used convolutional neural network architecture suitable for medical image segmentation tasks. It utilizes an encoder-decoder structure with skip connections to extract image features and restore spatial information.

### 2. **Enhanced U-Net**

Building on the basic U-Net, I added residual modules and more convolutional layers to further enhance the model's performance. The Enhanced U-Net model is better at capturing complex image features.

### 3. **FCN (Fully Convolutional Network)**

FCN is a convolutional neural network used for semantic segmentation. It replaces fully connected layers with fully convolutional layers to handle input images of arbitrary sizes.

### 4. **Attention U-Net**

Attention U-Net introduces attention mechanisms, enabling the model to better focus on important regions of the image, thereby improving segmentation accuracy, especially in images with complex backgrounds.

## Model Training

I used TensorFlow and Keras frameworks to build and train these models. The training process for each model is as follows:

- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy
- **Learning Rate**: Tuned based on the model and experimental results
- **Batch Size**: 1 or 2, depending on GPU memory constraints
- **Epochs**: 30

During training, I also used Keras callback functions to monitor and save the best models.

## Model Evaluation and Results

I used cross-entropy loss and accuracy as the primary evaluation metrics. Additionally, I calculated the Dice coefficient and IoU (Intersection over Union) for each model to better understand their performance on the segmentation task.

The test results for each model are as follows:

- **U-Net**: Test accuracy of 92.52%, average IoU of 0.725.
- **Enhanced U-Net**: Test accuracy of 85.73%, average Dice coefficient of 0.740.
- **FCN**: Test accuracy of 85.44%.
- **Attention U-Net**: Test accuracy of 90.33%, test loss of 0.226.

## Visualization

I provided detailed visualizations for the prediction results of each model, including raw images, ground truth labels, predicted labels, and enhanced binary predictions. These visualizations help provide a more intuitive understanding of the models' performance.

## Conclusion

Through this project, I gained a deeper understanding of various deep learning models' performance in image segmentation tasks. By utilizing different model architectures and techniques, I enhanced the segmentation capabilities of the models. Future work could include experimenting with more advanced data augmentation techniques and further optimizing the models for specific datasets.

