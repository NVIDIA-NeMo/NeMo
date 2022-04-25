Speech Classification
==================================
Speech Classification refers to a set of tasks or problems of getting a program to automatically classify input utterance or audio segment into categories, 
such as Speech Command Recognition (multi-class), Voice Activity Detection (binary or multi-class), and Audio Sentiment Classification (typically multi-class), etc.

**Speech Command Recognition** is the task of classifying an input audio pattern into a discrete set of classes. 
It is a subset of Automatic Speech Recognition, sometimes referred to as Key Word Spotting, in which a model is constantly analyzing speech patterns to detect certain "command" classes. 
Upon detection of these commands, a specific action can be taken by the system. 
It is often the objective of command recognition models to be small and efficient so that they can be deployed onto low-power sensors and remain active for long durations of time.


**Voice Activity Detection (VAD)** also known as speech activity detection or speech detection, is the task of predicting which parts of input audio contain speech versus background noise.
It is an essential first step for a variety of speech-based applications including Automatic Speech Recognition. 
It serves to determine which samples to be sent to the model and when to close the microphone.


The full documentation tree is as follows:

.. toctree::
   :maxdepth: 8

   models
   datasets
   results
   configs
   resources.rst

.. include:: resources.rst
