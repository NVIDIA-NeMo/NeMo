Speaker Recognition (SR)
========================

Speaker recognition is a broad research area which solves two major tasks: speaker identification (what is the identity of the speaker?) and speaker verification (is the speaker who they claim to be?). We focus on text-independent speaker recognition when the identity of the speaker is based on how the speech is spoken, not necessarily in what is being said. Typically such speaker recognition systems operate on unconstrained speech utterances, which are converted into vectors of fixed length, called speaker embeddings. Speaker embeddings can also be used in automatic speech recognition (ASR) and speech synthesis.

The goal of most speaker recognition systems is to get good speaker level representations that could help distinguish oneself from other speakers. To achieve this, we first train a neural network model in an end-to-end manner optimizing the encoder using cross-entropy or angular softmax loss. We modify the decoder to get these fixed size embeddings irrespective of the length of the audio input and employ a pooling strategy such as mean and variance based statistics pooling or attention based method to generate these embeddings.

In speaker identification, we typically train on a larger training set with cross-entropy loss and fine-tune later on preferred set of labels where one would want to classify only known sets of speakers. 
On the other hand, in speaker verification, we train an embedding extractor with angular softmax loss and compare the embeddings from one audio file coming from a single speaker with embeddings from an unknown speaker. For quantifying the similarity of the embeddings we use scoring techniques such as cosine similarity.

The full documentation tree: 

.. toctree::
   :maxdepth: 8

   models
   configs
   datasets
   results
   api
   resources

.. include:: resources.rst