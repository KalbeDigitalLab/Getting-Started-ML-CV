# Getting Started into Deep Learning for Computer Vision

This project will be a guideline for you to get started with deep learning. It will explain step-by-step from data analysis, data preprocessing, data augmentation pipeline, building model training pipeline using PyTorch, tracking the experiments with W&B, and finally deploying the model using FastAPI.

## Introduction
Deep learning has revolutionized the field of computer vision, enabling computers to understand and interpret visual data with remarkable accuracy. This project aims to provide a beginner-friendly approach to delve into the world of deep learning for computer vision. By following the outlined steps, you will gain hands-on experience in developing a simple classification model using the Food-101-tiny dataset.

Let's dive into the world of deep learning for computer vision and develop a simple classification model using the Food-101-tiny dataset!

## Dataset

For this project, we will be using the [Food-101-tiny](https://www.kaggle.com/datasets/msarmi9/food101tiny) dataset. It is a collection of images categorized into 101 different food classes. The dataset provides a diverse range of food images for classification tasks. The original [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) dataset contains images for 101 different classes of foods. Each class has 750 training images and 250 test images.

The original purpose of this dataset was to speed development and prototyping of an image classifier on the full Food-101 dataset and was used to experiment with a variety of techniques, including progressive resizing, label smoothing, and mixup. Training times should be under 30 seconds per epoch when using an image size of 224 or 256 and under a minute when using the max image size 512. We hope it will help anyone who's looking to quickly test out new ideas for improving performance on the original Food-101 dataset.

This dataset is a subset of Food-101. It contains 150 training images and 50 validation images for 10 easily classified foods: **apple pie**, **bibimbap**, **cannoli**, **edamame**, **falafel**, **french toast**, **ice cream**, **ramen**, **sushi**, and **tiramisu**.

Please ensure you have downloaded and set up the dataset before proceeding with the project.

```
KAGGLE_USERNAME=kaggle_username KAGGLE_KEY=kaggle_secret_key sh download_dataset.sh
```

Let's dive into the exciting world of deep learning for computer vision and develop a powerful food classification model together!

## Project Overview

The project consists of the following stages:

1. **Data Analysis**: Exploring and analyzing the dataset to gain insights.
2. **Data Preprocessing**: Preparing the data for training the model.
3. **Data Augmentation** Pipeline: Implementing techniques to augment the dataset.
4. **Building Model Training Pipeline**: Constructing a pipeline to train the model using PyTorch.
5. **Tracking Experiments with W&B**: Monitoring and tracking the experiments using W&B.
6. **Deploying the Model using FastAPI**: Deploying the trained model using FastAPI.

## Folder Structure

The project includes the following folders:

1. **Data Preprocessing**: Contains notebooks for data analysis and preprocessing.
2. **Model Development**: Contains notebooks for building and training the model.
3. **Model Deployment**: Contains notebooks for deploying the model using FastAPI.

Each folder will have several notebooks, each corresponding to a specific step in the process.
