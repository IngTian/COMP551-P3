# COMP551 Project 3 (Fall 2021)

## Acknowledgements

This repository is for **COMP551** at **McGill University** for **fall 2021**. Thanks to Zeying Tian, Yuyan Chen, and Yifei Chen, without whom this project cannot be accomplished.

## Description
In this project, we aim at predicting handwritten English letters and digits in an image. In this study, we have improved the performance progressively from the baseline VGG16 model. Concretely, we have experimented with multiple variants of VGG16 and two preprocessing techniques, namely data augmentation, and noise reduction. Also, we have tried to utilize the unlabeled training set with Expectation-Maximization (EM) algorithm. In the end, we have reached a validation accuracy of 97.1%.


## Project Structure

```console
├── dataset
│   ├── images_l.pkl
│   ├── images_test.pkl
│   ├── images_ul.pkl
│   └── labels_l.pkl
├── init.py
├── vgg16_vanilla.json
├── p3.ipynb # This is where you should have your colab file.
├── model
│   ├── AlexNetVanilla.py
│   ├── Baseline.py
│   ├── VGG16Vanilla.py
│   ├── VGG16_Deeper.py
│   ├── VGG16_Delete_Tail.py
│   ├── VGG16_Half_Width.py
│   ├── VGG16_Higher_MaxPooling.py
│   ├── VGG16_Higher_MaxPooling_Leaky.py
│   └── __init__.py
├── out
│   ├── records.json # The experiment recrods.
│   └── models
│       └── ...      # Various saved models.
├── out1 # For EM Records.
│   └── ...
├── requirements.txt
└── utils
    └── __init__.py
```
