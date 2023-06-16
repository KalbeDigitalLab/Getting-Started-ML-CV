# Getting Started into Machine Learning Cycle for Computer Vision Models

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
[![Lightning](https://img.shields.io/badge/Lightning-792ee5?style=for-the-badge&logo=pytorchlightning&logoColor=white)](https://www.pytorchlightning.ai/index.html)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://github.com/tiangolo/fastapi)
[![WeightAndBiases](https://img.shields.io/badge/W&B-F6C915?style=for-the-badge&logo=weightsandbiases&logoColor=black)](https://wandb.ai/)
[![ONNX](https://img.shields.io/badge/ONNX-A6A9AA?style=for-the-badge&logo=onnx&logoColor=white)](https://github.com/onnx/onnx)

<a href="https://hydra.cc/"><img align="center" src="https://raw.githubusercontent.com/facebookresearch/hydra/53d07f56a272485cc81596d23aad33e18e007091/website/static/img/Hydra-Readme-logo1.svg" height="30" style="background-color:white"></a> &nbsp; <a href="https://github.com/voxel51/fiftyone"><img align="center" src="https://github.com/voxel51/fiftyone/blob/develop/docs/source/_static/images/voxel51_300dpi.png?raw=true" height="30" style="background-color:white">&nbsp;<img align="center" src="https://github.com/voxel51/fiftyone/blob/develop/docs/source/_static/images/fiftyone.png?raw=true" height="30" style="background-color:white"></a>

---

![mlops-cycle](https://learn.microsoft.com/en-us/azure/architecture/example-scenario/mlops/media/data-sciene-lifecycle-model-flow.png)

This project will be a guideline for you to get started with machine learning cycle from development to production. It will explain step-by-step from data analysis, data preprocessing, data augmentation pipeline, building model training pipeline using PyTorch Lightning, tracking the experiments with W&B, and finally deploying the model using FastAPI.

*Altough this guideline explains about computer vision, but it is also aplicable to other modalitites.*

## Introduction
Deep learning has revolutionized the field of computer vision, enabling computers to understand and interpret visual data with remarkable accuracy. This project aims to provide a beginner-friendly approach to explore into the deep learning for computer vision. By following the outlined steps, you will gain hands-on experience in developing a simple classification model using the **Food-101-tiny** dataset.

## Project Overview

The project consists of the following stages:

1. **Data Analysis**: Exploring and analyzing the dataset to gain insights.
2. **Data Preprocessing and Augmentation**: Preparing the data and Implementing techniques augment the dataset.
3. **Data Visualization using Fiftyone**: Using fiftyone to visualize dataset.
4. **Simple Lightning Pipeline**: Constructing a pipeline to train the model using PyTorch Lightning.
5. **Tracking Experiments with W&B**: Monitoring and tracking the experiments using W&B.
6. **Advanced Training Pipeline**: Advanced method to run multiple experiments and hyperparameter search using Hydra and Optuna.
7. **Model Optimization**: Optimizing model latency using ONNX.
8. **Deploying the Model using Gradio**: Deploying model with simple interface using Gradio.
9. **Deploying the Model using FastAPI**: Deploying model as a web app using FastAPI.

## Folder Structure

The project includes the following folders:

1. **src**: Contains model development scripts, e.g. model, dataset, and utilities
2. **configs**: Configuration files to run lightning pipeline using Hydra
3. **pretrained**: Pretrained model for inference

Each folder will have several notebooks, each corresponding to a specific step in the process.

## Dataset

![food-101-dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/static/img/food-101.jpg)

For this project, we will be using the [Food-101-tiny](https://www.kaggle.com/datasets/msarmi9/food101tiny) dataset. It is a collection of images categorized into 101 different food classes. The dataset provides a diverse range of food images for classification tasks. The original [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) dataset contains images for 101 different classes of foods. Each class has 750 training images and 250 test images.

The original purpose of this dataset was to speed development and prototyping of an image classifier on the full Food-101 dataset and was used to experiment with a variety of techniques, including progressive resizing, label smoothing, and mixup. Training times should be under 30 seconds per epoch when using an image size of 224 or 256 and under a minute when using the max image size 512. We hope it will help anyone who's looking to quickly test out new ideas for improving performance on the original Food-101 dataset.

**Food-101-tiny** only contains 150 training images and 50 validation images for 10 easily classified foods: **apple pie**, **bibimbap**, **cannoli**, **edamame**, **falafel**, **french toast**, **ice cream**, **ramen**, **sushi**, and **tiramisu**.

Please ensure you have downloaded and set up the dataset before proceeding with the project.

```
KAGGLE_USERNAME=kaggle_username KAGGLE_KEY=kaggle_secret_key sh download_dataset.sh
```

Let's dive into the exciting world of deep learning for computer vision and develop a powerful food classification model together!
