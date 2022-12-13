# Transfer Learning with TensorFlow

All the machine learning problems can be solved by creating neural networks from scratch using our own data to fit in, compile and build and then further improve our network by adding more layers, adjusting the number of neurons, changing the learning rate, getting more data to train.

However, doing this will be very time consuming, specially if we have less data to train our model.

But we have the concept of **Transfer Learning**, which is like taking the patterns(also called weights) another model has learned from another problem and using them for our own problem.

There are two main benefits to using transfer learning,
1. Can leverage an existing neural network architecture **proven to work** on problems similar to our own.
2. Can leverage a working neural network architecture which has **already learned** patterns on similar data to our own. This often results in achieving great results with less custom data.

This means, instead of building our neural network architectures from scratch, we can utilise models which have worked for others.

**By the way, those models are trained on millions of custom data before getting publicise.**

## What we're going to cover

- Introduce Transfer Learning (a way to beat all of our old self-built models)
- Using a smaller dataset to experiment faster (10% of training samples of 10 classes of food)
- Build a transfer learning feature extraction model using TensorFlow Hub
- Introduce the TensorBoard callback to track model training results
- Compare model results using TensorBoard

## Exercises

- [x] Build and fit a model using the same data we have here but with the MobileNetV2 architecture feature extraction ([mobilenet_v2_100_224/feature_vector](https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4)) from TensorFlow Hub, how does it perform compared to our other models?
- [ ] Name 3 different image classification models on TensorFlow Hub that we haven't used.
- [ ] Build a model to classify images of two different things you've taken photos of.
    - You can use any feature extraction layer from TensorFlow Hub you like for this.
    - You should aim to have at least 10 images of each class, for example to build a fridge versus oven classifier, you'll want 10 images of fridges and 10 images of ovens.
- [ ] What is the current best performing model on ImageNet?
  - *Hint*: you might want to check [sotabench.com](https://www.sotabench.com/) for this.
  
## Extra Curriculam

- [ ] Read through the [TensorFlow Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning) and define the main two types of transfer learning in your own words.
- [ ] Go through the [Transfer Learning with TensorFlow Hub tutorial](https://www.tensorflow.org/tutorials/images/transfer_learning_with_hub) on the TensorFlow website and rewrite all of the code yourself into a new Google Colab notebook making comments about what each step does along the way.
- [ ] We haven't covered fine-tuning with TensorFlow Hub in this notebook, but if you'd like to know more, go through the [fine-tuning a TensorFlow Hub model tutorial](https://www.tensorflow.org/hub/tf2_saved_model#fine-tuning) on the TensorFlow homepage.How to fine-tune a tensorflow hub model:
- [ ] Look into [experiment tracking with Weights & Biases](https://www.wandb.com/experiment-tracking), how could you integrate it with our existing TensorBoard logs?
