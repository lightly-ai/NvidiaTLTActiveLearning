# Active Lerarning with the Nvidia TLT
Tutorial on active learning with the Nvidia Transfer Learning Toolkit (TLT).

1. [Prerequisites](#prerequisites)
    1. [Set up Lightly](#lightly)
    2. [Set up Nvidia TLT](#tlt)
    3. [Data](#data)
2. [Active Learning](#al)
    1. [Initial Sampling](#sampling)
    2. [Training and Inference](#training)
    3. [Active Learning Step](#alstep)
    4. [Re-training](#retraining)


## 1 Prerequisites <a name=prerequisites>

We start by creating a few directories which we'll need during the tutorial.
```
mkdir -p data/raw/images
mkdir -p data/raw/labels
```

Additionally, we need to install `lightly`, `numpy` and `matplotlib`.
```
pip install -r requirements.txt
```


### 1.1 Set up Lightly <a name=lightly>
To set up lightly, head to the [Lightly web-app](https://app.lightly.ai) and create a free account by logging in. Make sure to get your token by clicking on your e-mail address and selecting "Preferences". You will need the token for the rest of this tutorial.

### 1.2 Set up Nvidia TLT <a name=tlt>
TODO.

### 1.3 Data <a name=data>
We will use the [MinneApple fruit detection dataset](TODO). It consists of 670 training images of apple trees, annotated for detection and segmentation. The dataset contains images of trees with red and green apples.
TODO download!

## 2 Active Learning <a name=al>
Now that our setup is complete, we can start the active learning loop. In general, the active learning loop will consist of the following steps:
1. Initial sampling: Get an initial set of images to annotate and train on.
2. Training and inference: Train on the labeled data and make predictions on all data.
3. Active learning query: Use the predictions to get the next set of images to annotate, to to 2.

We will walk you through all three steps in this tutorial.

To do active learning with Lightly, we need to upload our dataset to the platform. The command `lightly-magic` will train a self-supervised model to get good image representations and then uploads the images along with the image representations to the platform. If you want to skip training, you can set `trainer.max_epochs=0`. In the following command, replace `MY_TOKEN` in the example with your own token from the platform.


```
lightly-magic \
    input_dir=./data/raw/images \
    trainer.max_epochs=0 \
    loader.num_workers=8 \
    collate.input_size=512 \
    new_dataset_name="MinneApple" \
    token=MY_TOKEN
```

The above command will display the id of your dataset. You will need this later in the tutorial.

Once the upload has finished, you can visually explore your dataset in the web-app.

<img src="./docs/gifs/MinneApple Lightly Showcase.gif">


### 2.1 Initial Sampling <a name=sampling>

Let's select an initial batch of images which we want to annotate.

Lightly offers different sampling strategies, the most prominent ones being `CORESET` and `RANDOM` sampling. `RANDOM` sampling will preserve the underlying distribution of your dataset well while `CORESET` maximizes the heterogeneity of your dataset. While exploring our dataset in the [web-app](https://app.lightly.ai), we notices many different clusters therefore we choose `CORESET` sampling to make sure that every cluster is represented in the training data.

We use the `active_learning_query.py` script to make an initial selection:

```
python active_learning_query.py TODO
```

The above script roughly performs the following steps:
TODO

### 2.2 Training and Inference <a name=training>
Now that we have our annotated training data, let's train an object detection model on it and see how well it works! We use the Nvidia Transfer Learning Toolkit which allows us to train a YOLOv4 object detector from the command line. The cool thing about transfer learning is that we don't have to train a model from scratch and so we require fewer annotated images to get good results.

Let's start by downloading a pre-trained object detection model from the Nvidia registry.

```
mkdir -p ./yolo_v4/pretrained_resnet18
ngc registry model download-version nvidia tlt_pretrained_object_detection:resnet18 \
    --dest ./yolo_v4/pretrained_resnet18
```

Finetuning the object detector on our sampled training data is as simple as the following command. Make sure to replace MY_KEY with the API token you get from your Nvidia account TODO.

```
mkdir -p $PWD/yolo_v4/experiment_dir_unpruned
tlt yolo_v4 train \
    -e /workspace/tlt-experiments/yolo_v4/specs/yolo_v4_minneapple.txt \
    -r /workspace/tlt-experiments/yolo_v4/experiment_dir_unpruned \
    --gpus 1 \
    -k MY_KEY
``` 

Now that we have finetuned the object detector on our dataset, we can do inference to see how well it works.

Doing inference on the whole dataset has the advantage that we can figure out for which images the model performs poorly or has a lot of uncertainties.

```
tlt yolo_v4 inference \
    -i /workspace/tlt-experiments/data/raw/images/ \
    -e /workspace/tlt-experiments/yolo_v4/specs/yolo_v4_minneapple.txt \
    -m /workspace/tlt-experiments/yolo_v4/experiment_dir_unpruned/weights/yolov4_resnet18_epoch_050.tlt \
    -o /workspace/tlt-experiments/infer_images \
    -l /workspace/tlt-experiments/infer_labels \
    -k MY_KEY
```

Below you can see two example images after training.

<img src="./docs/examples/MinneApple_labeled_vs_unlabeled.png">


### 2.3 Active Learning Step <a name=alstep>
We can use the inferences from the previous step to determine with which images the model has problems. With Lightly, we can easily select these images while at the same time making sure that our training dataset is not flooded with duplicates.

This section is about how to select the images which complete your training dataset.

TODO

### 2.4 Re-training <a name=retraining>

We can re-train our object detector on the new dataset to get an even better model. For this, we can use the same command as before. If you want to continue training from the last checkpoint, make sure to replace the pretrain_model_path in the specs file by a resume_model_path:

```
tlt yolo_v4 train \
    -e /workspace/tlt-experiments/yolo_v4/specs/yolo_v4_minneapple.txt \
    -r /workspace/tlt-experiments/yolo_v4/experiment_dir_unpruned \
    --gpus 1 \
    -k MY_KEY
```