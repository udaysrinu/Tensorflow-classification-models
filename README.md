# Tensorflow-classification-models

A small portfolio of Keras/TensorFlow image-classification notebooks built while learning CNNs — covers MNIST, Fashion-MNIST, sign-language MNIST, happy/sad faces, cats vs dogs (with and without augmentation), horses vs humans (transfer learning), and an Inception-v3 transfer-learning notebook.

> Built 2020 as a learning portfolio. All notebooks open directly in Google Colab via the badge at the top of each.

## What it does

Each notebook trains a small Keras model on a public dataset and saves no checkpoints — they're meant as runnable lab exercises, not production models. Together they cover the standard 2020 "intro to CNNs" curriculum: dense baseline, conv baseline, augmentation, dropout, transfer learning.

## Notebooks

| Notebook | Task | Architecture | Reported result |
|---|---|---|---|
| [`MNIST_handwriting.ipynb`](MNIST_handwriting.ipynb) | MNIST digit classification | Dense baseline: Flatten -> Dense(512, ReLU) -> Dense(10, softmax) | 99.10% train acc at epoch 5 (early-stop callback at >99%) |
| [`MNiST_dataset_classification_with_convolution.ipynb`](MNiST_dataset_classification_with_convolution.ipynb) | MNIST with a single conv layer | Conv2D(32,3) -> MaxPool -> Flatten -> Dense(128) -> Dense(10) | 99.82% train acc at epoch 8 (early-stop at >99.8%) |
| [`Fashion_Mnist_data_classification_model.ipynb`](Fashion_Mnist_data_classification_model.ipynb) | Fashion-MNIST 10-way | Dense baseline (same shape as MNIST) | 83.02% train acc at epoch 1 (early-stop callback fires at >80%) |
| [`Binary_classification_between_happy_and_sad_faces.ipynb`](Binary_classification_between_happy_and_sad_faces.ipynb) | Happy/sad face binary classifier (Laurence Moroney "happy-or-sad" dataset, 80 imgs) | 3 Conv blocks (16/32/32 filters, 3x3) -> Flatten -> Dense(512) -> Dense(1, sigmoid), RMSprop | 100.00% train acc at epoch 11 — clearly overfits 80 images, more a callback exercise than a real result |
| [`Multi_class_classifier_MNIST sign dataset.ipynb`](Multi_class_classifier_MNIST%20sign%20dataset.ipynb) | Sign-language MNIST, 24-way | CNN with augmentation | See notebook |
| [`Cats_vs_Dogs_kaggle challenge.ipynb`](Cats_vs_Dogs_kaggle%20challenge.ipynb) | Kaggle cats-vs-dogs binary | CNN, no augmentation | See notebook |
| [`Cats_vs_Dogs_using_augmentation_kaggle challenge.ipynb`](Cats_vs_Dogs_using_augmentation_kaggle%20challenge.ipynb) | Same task, with augmentation | CNN + ImageDataGenerator augmentation | See notebook (the point of this one is to show the gap aug closes vs. the previous notebook) |
| [`Horses_vs_humans_using_Transfer_Learning.ipynb`](Horses_vs_humans_using_Transfer_Learning.ipynb) | Horses vs humans binary | Inception-v3 frozen backbone -> custom head | See notebook |
| [`Transfer_learning_with_inception_v3.ipynb`](Transfer_learning_with_inception_v3.ipynb) | Inception-v3 transfer learning playground | Frozen Inception backbone, custom dense head | See notebook |

Tactics shown across the set: early-stopping callbacks based on accuracy, RMSprop vs Adam, `binary_crossentropy` vs `sparse_categorical_crossentropy`, `ImageDataGenerator` for augmentation, `flow_from_directory`, dropout, transfer learning with a frozen backbone.

## Tech stack

- TensorFlow 2.x / `tf.keras` (Colab-default)
- Python 3 (Colab)
- Datasets: `tf.keras.datasets.mnist`, `tf.keras.datasets.fashion_mnist`, Laurence Moroney's `happy-or-sad`, Kaggle cats-vs-dogs, Sign-MNIST, horses-vs-humans
- Inception-v3 weights via `tf.keras.applications`

## How to run

Easiest path: open any notebook in Colab using the "Open In Colab" badge at the top of each `.ipynb`. Locally:

```
pip install tensorflow jupyter matplotlib pillow
jupyter notebook
```

The notebooks `wget` their datasets at runtime, so you don't need to pre-download anything (assumes internet).

## Honest take

These are learning exercises, not benchmarks. The MNIST conv result (99.82%) and dense MNIST (99.10%) match what every other intro-CNN notebook on the internet gets. The happy/sad 100% is on 80 images and overfits — the callback (`stop training at 99.9%`) is the actual learning objective there. Where these are useful is reading the diff between the cats-vs-dogs notebook with and without augmentation — that's where the technique earns its keep.

## License

No license file was included with the original project. Treat as "all rights reserved, ask before reuse" until I add one. Datasets used belong to their original providers (Google, Kaggle, Laurence Moroney).
