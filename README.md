# Plant Disease Classification with Hybrid Model (VGG19 + ResNet)

This repository contains a hybrid plant disease classification model using VGG19 and ResNet architectures, integrated with a Flask web application. The model predicts whether a plant is infected or not based on the input image.

## Features

- **Hybrid Model**: Uses a combination of VGG19 and ResNet architectures for improved plant disease classification accuracy.
- **Flask Web Application**: A simple web interface where users can upload an image of a plant leaf and get a prediction on whether the plant is infected or healthy.
- **Image Classification**: The model classifies plant leaves into two categories: infected or healthy. 

## Installation

Follow the steps below to set up and run the application:

### Prerequisites

1. Python 3.7 or higher
2. pip (Python package installer)

## Model Overview

The hybrid model used in this project combines the strengths of VGG19 and ResNet to improve accuracy in plant disease classification. By leveraging the power of these pre-trained deep learning models, we can achieve higher performance on plant leaf images, detecting infections caused by various plant diseases.

## Model Workflow

1. **Image Preprocessing**: The uploaded plant image is resized and normalized for model input.
2. **Model Prediction**: The image is passed through the hybrid model, which is composed of layers from both VGG19 and ResNet.
3. **Result Output**: The model outputs a classification label indicating whether the plant is infected or healthy.
