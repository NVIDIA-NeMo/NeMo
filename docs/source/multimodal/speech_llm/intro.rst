Speech-agumented Large Language Models (SpeechLLM)
==================================================

The endeavor to extend Language Models (LLMs) with the ability to understand speech and audio inputs, detailed examples can be found in the `SpeechLLM example <https://github.com/NVIDIA/NeMo/blob/main/examples/multimodal/speech_llm/README.md>`_.. 

.. toctree::
   :maxdepth: 1
   datasets
   configs
   api


In general, there're three main components of a modular SpeechLLM: 
- An audio encoder that processes the input audio and produces a sequence of audio embeddings.
- A modality adapter that processes the audio embeddings and produces a sequence of embeddings in the same latent space as the token embeddings of a pretrained large language model (LLM).
- A pretrained large language model (LLM) that processes embeddings from the modality adapter as well as token embeddings of input prompt, and produces the text output. The audio embeddings and text token embeddings are concatenated in time dimension before going into the LLM.
- The LLM produces text outputs based on the concatenated input audio and text embedding.


Model Architecture
^^^^^^^^^^^^^^^^^^

One way to incorporate speech into LLM is to concatenate speech features with the token embeddings of the input text prompt before being fed into the LLM. In this way, the LLM can have direct access to the speech information when generating the output text.
    .. image:: https://github.com/NVIDIA/NeMo/releases/download/v1.23.0/salm.png
        :align: center
        :alt: SALM model
        :scale: 50%



Another way is to use cross-attention mechanism, by using text embeddings to attend to speech embeddings to extract task-specific information from the speech embeddings. In order to minimize the computational cost of cross-attention, we add a cross-attention module only before the LLM.
   
    .. image:: https://github.com/NVIDIA/NeMo/releases/download/v1.23.0/bestow.png
        :align: center
        :alt: BESTOW model
        :scale: 50%





