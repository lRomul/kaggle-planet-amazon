Planet: Understanding the Amazon from Space
===========================================

Use satellite data to track the human footprint in the Amazon rainforest.

This is my part of our team's solution for the Kaggle challange of
[Understanding the Amazon from Space](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space).

Our team ods.ai [finished 7th](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/leaderboard/private).

## Requirements
* Linux
* Nvidia drivers, CUDA 8
* Docker, nvidia-docker

## How to Use?

Put data to ``data``::

    data
    ├── train-jpg
    ├── test
    ├── 10_folds.npy
    ├── train_v2.csv
    └── sample_submission_v2.csv

`test` contains images from `test-jpg` and `test-jpg-additional`.

1. Go to folder `docker` and build image
```
cd docker
./build.sh
```

2. Run container with nvidia-docker
```
./run.sh
```

3. Go to folder `src/kfold_train` and start train models

```
cd src/kfold_train/
python densenet121_001.py
python vgg11_001.py
...
python vgg19_001.py
```