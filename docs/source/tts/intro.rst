Speech Synthesis (TTS)
======================

.. toctree::
   :maxdepth: 4
   :hidden:

   datasets
   api

Speech Synthesis or Text-to-Speech (TTS) involves turning text into human speech. The NeMo TTS collection currently
supports two pipelines for TTS:

1) The two stage pipeline. First, a model is used to generate a mel spectrogram from text. Second, a model is used
to generate audio from a mel spectrogram.
2) The "end-to-end" approach that uses one model to generate audio straight from text.

Quick Start:

.. code-block:: python

    import soundfile as sf
    from nemo.collections.tts.models.base import SpectrogramGenerator, Vocoder

    # Download and load the pretrained fastpitch model
    spec_generator = SpectrogramGenerator.from_pretrained(model_name="tts_en_fastpitch").cuda()
    # Download and load the pretrained hifigan model
    vocoder = Vocoder.from_pretrained(model_name="tts_hifigan").cuda()

    # All spectrogram generators start by parsing raw strings to a tokenized version of the string
    parsed = spec_generator.parse("You can type your sentence here to get nemo to produce speech.")
    # They then take the tokenized string and produce a spectrogram
    spectrogram = spec_generator.generate_spectrogram(tokens=parsed)
    # Finally, a vocoder converts the spectrogram to audio
    audio = vocoder.convert_spectrogram_to_audio(spec=spectrogram)

    # Save the audio to disk in a file called speech.wav
    # Note vocoder return a batch of audio. In this example, we just take the first and only sample.
    sf.write("speech.wav", audio.to('cpu').detach().numpy()[0], 22050)

.. note::

   For an interactive version of the quick start above, refer to the TTS inference notebook that can be found on the
   github readme.

Available Models
################

NeMo supports a variety of models that can be used for TTS. For beginners, we recommend starting with the FastPitch +
HiFiGAN combination. Then we suggest trying using FastPitch_HifiGan_E2E if you want to start exploring
the end-to-end pipeline.

.. list-table:: *TTS Models*
   :widths: 5 5 10 25
   :header-rows: 1

   * - Model
     - Base Class
     - Pretrained Checkpoint
     - Description
   * - Tacotron2
     - :class:`SpectrogramGenerator<nemo.collections.tts.models.base.SpectrogramGenerator>`
     - https://ngc.nvidia.com/catalog/models/nvidia:nemo:tts_en_tacotron2
     - LSTM encoder decoder based model that generates spectrograms
   * - GlowTTS
     - :class:`SpectrogramGenerator<nemo.collections.tts.models.base.SpectrogramGenerator>`
     - https://ngc.nvidia.com/catalog/models/nvidia:nemo:tts_en_glowtts
     - Glow-based spectrogram generator
   * - FastSpeech 2
     - :class:`SpectrogramGenerator<nemo.collections.tts.models.base.SpectrogramGenerator>`
     - https://ngc.nvidia.com/catalog/models/nvidia:nemo:tts_en_fastspeech_2
     - Non-autoregressive transformer-based spectrogram generator that predicts duration, energy, and pitch
   * - FastPitch
     - :class:`SpectrogramGenerator<nemo.collections.tts.models.base.SpectrogramGenerator>`
     - https://ngc.nvidia.com/catalog/models/nvidia:nemo:tts_en_fastpitch
     - Non-autoregressive transformer-based spectrogram generator that predicts duration and pitch
   * - TalkNet
     - :class:`SpectrogramGenerator<nemo.collections.tts.models.base.SpectrogramGenerator>`
     - https://ngc.nvidia.com/catalog/models/nvidia:nemo:tts_en_talknet
     - Non-autoregressive convolution-based spectrogram generator
   * - Mixer-TTS
     - :class:`SpectrogramGenerator<nemo.collections.tts.models.base.SpectrogramGenerator>`
     - https://ngc.nvidia.com/catalog/models/nvidia:nemo:tts_en_lj_mixertts
     - Non-autoregressive mixer-based spectrogram generator
   * - Mixer-TTS-X
     - :class:`SpectrogramGenerator<nemo.collections.tts.models.base.SpectrogramGenerator>`
     - https://ngc.nvidia.com/catalog/models/nvidia:nemo:tts_en_lj_mixerttsx
     - Non-autoregressive mixer-based spectrogram generator conditioned on language model embeddings
   * - WaveGlow
     - :class:`Vocoder<nemo.collections.tts.models.base.Vocoder>`
     - https://ngc.nvidia.com/catalog/models/nvidia:nemo:tts_waveglow_88m
     - Glow-based vocoder
   * - SqueezeWave
     - :class:`Vocoder<nemo.collections.tts.models.base.Vocoder>`
     - https://ngc.nvidia.com/catalog/models/nvidia:nemo:tts_squeezewave
     - Glow-based vocoder based on WaveGlow but with fewer parameters
   * - UniGlow
     - :class:`Vocoder<nemo.collections.tts.models.base.Vocoder>`
     - https://ngc.nvidia.com/catalog/models/nvidia:nemo:tts_uniglow
     - Glow-based vocoder based on WaveGlow but shares 1 set of parameters across all flow steps
   * - MelGAN
     - :class:`Vocoder<nemo.collections.tts.models.base.Vocoder>`
     - https://ngc.nvidia.com/catalog/models/nvidia:nemo:tts_melgan
     - GAN-based vocoder
   * - HiFiGAN
     - :class:`Vocoder<nemo.collections.tts.models.base.Vocoder>`
     - https://ngc.nvidia.com/catalog/models/nvidia:nemo:tts_hifigan
     - GAN-based Vocoder
   * - UnivNet
     - :class:`Vocoder<nemo.collections.tts.models.base.Vocoder>`
     - https://ngc.nvidia.com/catalog/models/nvidia:nemo:tts_en_lj_univnet
     - GAN-based vocoder
   * - FastPitch_HifiGan_E2E
     - :class:`TextToWaveform<nemo.collections.tts.models.base.TextToWaveform>`
     - https://ngc.nvidia.com/catalog/models/nvidia:nemo:tts_en_e2e_fastpitchhifigan
     - End-to-end model based on composing FastPitch and HiFiGAN
   * - FastSpeech2_HifiGan_E2E
     - :class:`TextToWaveform<nemo.collections.tts.models.base.TextToWaveform>`
     - https://ngc.nvidia.com/catalog/models/nvidia:nemo:tts_en_e2e_fastspeech2hifigan
     - End-to-end model based on composing FastSpeech2 and HiFiGAN

Base Classes
############

Two Stage Pipeline
******************

NeMo TTS has two base classes corresponding to the two stage pipeline:

  - :class:`SpectrogramGenerator<nemo.collections.tts.models.base.SpectrogramGenerator>`
  - :class:`Vocoder<nemo.collections.tts.models.base.Vocoder>`

The :class:`SpectrogramGenerator<nemo.collections.tts.models.base.SpectrogramGenerator>` class has two important
functions: :py:meth:`parse<nemo.collections.tts.models.base.SpectrogramGenerator.parse>` which
accepts raw python strings and returns a torch.tensor that represents tokenized text ready to pass to, and
:py:meth:`generate_spectrogram<nemo.collections.tts.models.base.SpectrogramGenerator.generate_spectrogram>` which
accepts a batch of tokenized text and returns a torch.tensor that represents a batch of spectrograms

The :class:`Vocoder<nemo.collections.tts.models.base.Vocoder>` class has one important functions
:py:meth:`convert_spectrogram_to_audio<nemo.collections.tts.models.base.Vocoder.convert_spectrogram_to_audio>` which
accepts a batch of spectrograms and returns a torch.tensor that represents a batch of raw audio.

End-to-End Pipeline
*******************

Correspondingly, NeMo TTS has one base class for the end-to-end pipeline:

  - :class:`TextToWaveform<nemo.collections.tts.models.base.TextToWaveform>`

Similarly to the SpectrogramGenerator, :class:`TextToWaveform<nemo.collections.tts.models.base.TextToWaveform>`
implements two functions: :py:meth:`parse<nemo.collections.tts.models.base.TextToWaveform.parse>` which
accepts raw python strings and returns a torch.tensor that represents tokenized text ready to pass to, and
:py:meth:`generate_spectrogram<nemo.collections.tts.models.base.TextToWaveform.convert_text_to_waveform>` which
accepts a batch of tokenized text and returns a torch.tensor that represents a batch of audio.

TTS Training
############

Training of TTS models can be done using the scripts inside the NeMo ``examples/tts folders``. The majority of the TTS
YAML configurations should work out of the box with the LJSpeech dataset. If you want to train on other data, it is
recommended that you walk through the Tacotron 2 Training notebook. It is applicable to most of NeMo's spectrogram
generator models. Please pay special attention to the sample rate and FFT parameters for new data.

Some models, like FastSpeech 2, require supplementary data such as phoneme durations, pitches, and energies.
For more information about how to preprocess datasets for such models, see the `TTS Datasets <./datasets.html>`__ page.
