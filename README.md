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

