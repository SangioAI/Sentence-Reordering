# Sentence Reordering

A light-weight (*5.86M* params) Transformer-based model in Keras 3 to reorder shuffled senteces.

# Results

It achieves *~43%* on the metric provided with a *~89%* accuracy on the training set. Full capicity of the model was reached in *25 epoch* of training.

The model score is *4.8* standard deviation away from baseline(random model) score.

# Dependencies

For this project is important to have:
- [Keras 3](https://keras.io/getting_started/)
- [Tensorflow 2.16](https://pypi.org/project/tensorflow/2.16.2/)

# Metric

Let s be the source string and p your prediction. The quality of the results will be measured according to the following metric:

1.  look for the longest substring w between s and p
2.  compute |w|/max(|s|,|p|)

If the match is exact, the score is 1.

When computing the score, the start and end tokens should NOT be considered.

# Notes

Here some notes on the choices made:

 - Previously, I tried a LSTM-based encoder-decoder sequence-2-sequence model, but learning was slower.

 - Larger models has not been tested due to limited resources available. The model capicity, and the score accordingly could have benefit from it. 

 - Stem preprocessing is not applied. The score, the computation time and the number of parameters for the two Embedding Layers could have benefit from it. Both because I tried it too late and it's not a completely reversible process, I din't use it.
 
 - First Residual connection of decoder was deleted, because it carried the target not masked to the following layers, making the network cheating. A solution could have been developing a custom layer able to apply a time distributed mask to that residual link. The backpropagation and score would have benefit from it, but I didn't have the time.
 
 - Other interesting approaches to the task:
    - Same transformer-based architecture, but using positional encoding on encoder inputs and training on unshuffled data only. After training, replace only the encoder with the very same one without positional encoding and, letting it be the only component trainable, train the model a second time on shuffle data. This second training could also be a regression task on the latent space projected by the first transformer-based architecture.
    - Diffusion-based architecture, requires a notion of randomness or noise of shuffled text data to be learnt.
