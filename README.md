# Transfer Learning with TensorFlow

All the machine learning problems can be solved by creating neural networks from scratch using our own data to fit in, compile and build and then further improve our network by adding more layers, adjusting the number of neurons, changing the learning rate, getting more data to train.

However, doing this will be very time consuming, specially if we have less data to train our model.

But we have the concept of **Transfer Learning**, which is like taking the patterns(also called weights) another model has learned from another problem and using them for our own problem.

There are two main benefits to using transfer learning,
1. Can leverage an existing neural network architecture **proven to work** on problems similar to our own.
2. Can leverage a working neural network architecture which has **already learned** patterns on similar data to our own. This often results in achieving great results with less custom data.

This means, instead of building our neural network architectures from scratch, we can utilise models which have worked for others.

**By the way, those models are trained on millions of custom data before getting publicise.**

# Types of Transfer Learning

0. **Feature Extraction** : Similar Architecture of Model but our own custom dataset
1. **Fine-Tuning** : Some layers are unfrozen to fine-tune and might need more data to train
2. **Use As It Is**

# 0. Feature Extraction

This section demonstrates how we use transfer learning for Feature Extraction.


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SaketMunda/transfer-learning-with-tensorflow/blob/master/transfer_learning_with_tensorflow_feature_extraction.ipynb)

## What we're going to cover

- Introduce Transfer Learning (a way to beat all of our old self-built models)
- Using a smaller dataset to experiment faster (10% of training samples of 10 classes of food)
- Build a transfer learning feature extraction model using TensorFlow Hub
- Introduce the TensorBoard callback to track model training results
- Compare model results using TensorBoard

## Exercises

- [x] Build and fit a model using the same data we have here but with the MobileNetV2 architecture feature extraction ([mobilenet_v2_100_224/feature_vector](https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4)) from TensorFlow Hub, how does it perform compared to our other models?
- [x] Name 3 different image classification models on TensorFlow Hub that we haven't used.
- [x] Build a model to classify images of two different things you've taken photos of.
    - You can use any feature extraction layer from TensorFlow Hub you like for this.
    - You should aim to have at least 10 images of each class, for example to build a fridge versus oven classifier, you'll want 10 images of fridges and 10 images of ovens.
- [x] What is the current best performing model on ImageNet?
  - *Hint*: you might want to check [sotabench.com](https://www.sotabench.com/) for this.
  
## Extra Curriculam

- [ ] Read through the [TensorFlow Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning) and define the main two types of transfer learning in your own words.
- [ ] Go through the [Transfer Learning with TensorFlow Hub tutorial](https://www.tensorflow.org/tutorials/images/transfer_learning_with_hub) on the TensorFlow website and rewrite all of the code yourself into a new Google Colab notebook making comments about what each step does along the way.
- [ ] We haven't covered fine-tuning with TensorFlow Hub in this notebook, but if you'd like to know more, go through the [fine-tuning a TensorFlow Hub model tutorial](https://www.tensorflow.org/hub/tf2_saved_model#fine-tuning) on the TensorFlow homepage.How to fine-tune a tensorflow hub model:
- [ ] Look into [experiment tracking with Weights & Biases](https://www.wandb.com/experiment-tracking), how could you integrate it with our existing TensorBoard logs?

# 1. Fine Tuning

This section demonstrates how we use transfer learning for Fine Tuning.

In **fine-tuning transfer learning** the pretrained model weights from another model are unfrozen and tweaked during to better suit our own data.

For feature extraction transfer learning, you may only train the top 1-3 layers like adjust the input layer of a pretrained model with your own data, in fine-tuning transfer learning, you might train 1-3+ layers of a pre-trained model (where the '+' indicates that many or all of the layers could be trained).

## What we're going to cover

We're going to go through the follow with TensorFlow:

- Introduce fine-tuning, a type of transfer learning to modify a pre-trained model to be more suited to your data
- Using the Keras Functional API (a differnt way to build models in Keras)
- Using a smaller dataset to experiment faster (e.g. 1-10% of training samples of 10 classes of food)
- Data augmentation (how to make your training dataset more diverse without adding more data)
- Running a series of modelling experiments on our Food Vision data
    - Model 0: a transfer learning model using the Keras Functional API
    - Model 1: a feature extraction transfer learning model on 1% of the data with data augmentation
    - Model 2: a feature extraction transfer learning model on 10% of the data with data augmentation
    - Model 3: a fine-tuned transfer learning model on 10% of the data
    - Model 4: a fine-tuned transfer learning model on 100% of the data
- Introduce the ModelCheckpoint callback to save intermediate training results
- Compare model experiments results using TensorBoard

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SaketMunda/transfer-learning-with-tensorflow/blob/master/fine_tuning_transfer_learning_with_tensorflow.ipynb)

## Exercises

- [x] Write a function to visualize an image from any dataset (train or test file) and any class (e.g. "steak", "pizza"... etc), visualize it and make a prediction on it using a trained model.
- [x] Use feature-extraction to train a transfer learning model on 10% of the Food Vision data for 10 epochs using [tf.keras.applications.EfficientNetB0](https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB0) as the base model. Use the [ModelCheckpoint](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint) callback to save the weights to file.
- [x] Fine-tune the last 20 layers of the base model you trained in 2 for another 10 epochs. How did it go?
- [x] Fine-tune the last 30 layers of the base model you trained in 2 for another 10 epochs. How did it go?


## Extra Curriculum

- [ ] Read the [documentation on data augmentation](https://www.tensorflow.org/tutorials/images/data_augmentation) in TensorFlow.
- [ ] Read the [ULMFit paper](https://arxiv.org/abs/1801.06146) (technical) for an introduction to the concept of freezing and unfreezing different layers.
- [ ] Read up on learning rate scheduling (there's a [TensorFlow callback](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/LearningRateScheduler) for this), how could this influence our model training?
    - If you're training for longer, you probably want to reduce the learning rate as you go... the closer you get to the bottom of the hill, the smaller steps you want to take. Imagine it like finding a coin at the bottom of your couch. In the beginning your arm movements are going to be large and the closer you get, the smaller your movements become.
    
# 2. Scaling Up

In this section, we're going to scale up from using 10 classes of the Food101 data to using all of the classes in the Food101 dataset.

Our goal is to beat the original [Food101 paper's](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/static/bossard_eccv14_food-101.pdf) results with 10% of data.

## What we're going to cover

- Downloading and preparing 10% of the Food101 data (10% of training data)
- Training a feature extraction transfer learning model on 10% of the Food101 training data
- Fine-tuning our feature extraction model
- Saving and loaded our trained model
- Evaluating the performance of our Food Vision model trained on 10% of the training data
    - Finding our model's most wrong predictions
- Making predictions with our Food Vision model on custom images of food

## Exercises

- [x] Take 3 of your own photos of food and use the trained model to make predictions on them, share your predictions with the other students in Discord and show off your Food Vision model 🍔👁.
- [x] Train a feature-extraction transfer learning model for 10 epochs on the same data and compare its performance versus a model which used feature extraction for 5 epochs and fine-tuning for 5 epochs (like we've used in this notebook). Which method is better?
- [x] Recreate the first model (the feature extraction model) with [`mixed_precision`](https://www.tensorflow.org/guide/mixed_precision) turned on.
    - Does it make the model train faster?
    - Does it effect the accuracy or performance of our model?
    - What's the advantages of using `mixed_precision` training?

## Extra-Curriculum

- [ ] Spend 15-minutes reading up on the [EarlyStopping callback](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping). What does it do? How could we use it in our model training?
- [ ] Spend an hour reading about [Streamlit](https://www.streamlit.io/). What does it do? How might you integrate some of the things we've done in this notebook in a Streamlit app?

# Resources

This curriculam and topics are learned from [Mr. D Bourke's tutorials of Deep Learning](https://dev.mrdbourke.com/tensorflow-deep-learning/04_transfer_learning_in_tensorflow_part_1_feature_extraction/) 
