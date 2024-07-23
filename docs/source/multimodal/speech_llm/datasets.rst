SpechLLM Dataset
================

The dataset classes can be found on `NeMo GitHub <https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/multimodal/speech_llm/data/audio_text_dataset.py>`_.


Input Manifest Format
---------------------

You'll need to prepare data in the NeMo manifest format, where each line is a python dictionary with some keys, for example:

.. code-block:: yaml

    {
        "audio_filepath": "path/to/audio.wav",
        "offset": 0.0, # offset of the audio in seconds, this is an optional field
        "duration": 10.0 , # duration of the audio in seconds, can set to `None` to load the whole audio
        "context": "what is the transcription of the audio?", # text prompt for the audio, see below for more details
        "answer": "the transcription of the audio", # optional for inference, default to "na" in dataloader
    }


The `context` field in the manifest is optional, and you can put a list of context in a context file (one context for each line) then set `++model.data.train_ds.context_file=<path to to context file>` to ask the dataloader to randomly pick a context from the file for each audio sample. This is useful for training with multiple prompts for the same task. If neither `context` field nor `context_file` is provided, the dataloader will use a default context `what does the audio mean?` for all audios. During inference, it is recommended to have the `context` field in the manifest. 

Customizing the fields to use
-----------------------------

You can also use other fields in the manifest to replace the `context` and `answer`fields, but you'll also need to change the `prompt_template` to use the new field names. For example, if you desire to use the new fields `input_text` and `output_text`, you need to set:

.. code-block:: bash

    ++model.data.train_ds.context_key=input_text \
    ++model.data.train_ds.answer_key=output_text \
    ++model.data.train_ds.prompt_template="'Q: {input_text}\nA: {output_text}'"

Note that there're single quotes around the prompt template (to avoid hydra errors), and the field names are wrapped in curly braces.


Customizing the input format
----------------------------

If you would like to use multiple audios, you can set the `audio_filepath` to be a list of audio file paths, and specify the location of each audio by using a special `audio_locator` string in the context. The choice of `audio_locator` should also be passed into the config. For example, if you have a manifest item like this:

.. code-block:: yaml

    {
        "audio_filepath": ["path/to/audio1.wav", "path/to/audio2.wav"],
        "context": "what is the transcription of the [audio] and [audio]?", # text prompt for the audio, see below for more details
        "answer": "the transcription of the audio1 and audio2", # optional for inference, default to "na" in dataloader
    }


You can set the `audio_locator` to be `[audio]` in the config:

.. code-block:: bash

    ++model.data.train_ds.audio_locator='[audio]'


By using `audio_locator`, the dataloader will replace the `audio_locator` in the context with the corresponding audio features extracted for each audio. You need to make sure that the number of audio locators in the context matches the number of audio files in the `audio_filepath` field. 



Multi-task Training
-------------------


In order to use a context file, you can set `++model.data.train_ds.context_file=<path to to context file>` in the command line or use multiple context files with `++model.data.train_ds.context_file=[<path to to context file1>,<path to context file2>,...]`. If the number of context files is equal to the number of provided datasets, the dataloader will assigne each context file to a dataset. Otherwise, the dataloader will randomly pick a context file from all provided context files for each audio sample. Using multiple context files is useful for training with multiple tasks, where each task has its own set of prompts. Meanwhile, you can control the weights for different tasks/datasets by using concatentated tarred datasets, where you can assign weights to datasets by:

.. code-block:: bash

    ++model.data.train_ds.is_tarred=True \
    ++model.data.train_ds.is_concat=True \
    ++model.data.train_ds.manifest_filepath=[/path/to/data1/tarred_audio_manifest.json,/path/to/data2/tarred_audio_manifest.json] \
    ++model.data.train_ds.tarred_audio_filepaths=[/path/to/data1/audio__OP_0..1023_CL_.tar,/path/to/data2/audio__OP_0..1023_CL_.tar] \
    ++model.data.train_ds.concat_sampling_technique='random' \
    ++model.data.train_ds.concat_sampling_probabilities=[0.4,0.6] \



Use Lhotse Dataloader
---------------------

Speech-LLM supports NeMo dataloader and Lhotse dataloader. Most of the Lhotse specific flags can be referred to `Lhotse Dataloader <https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/datasets.html#lhotse-dataloading>`.
Example config can be referred to `Lhotse Speech-LLM examples <https://github.com/NVIDIA/NeMo/blob/main/examples/multimodal/speech_llm/conf/salm/modular_audio_gpt_config_llama_lhotse.yaml>`_.

Lhotse Dataloader also supports using a standalone YAML file to set up the manifest info:

.. code-block:: bash

    ++model.data.train_ds.input_cfg=$INPUT_CFG_FILE \

which points to a $INPUT_CFG_FILE file like the following:

.. code-block:: yaml

    - input_cfg:
    - manifest_filepath: manifest1.json
        type: nemo
        weight: 2.0
        tags:
        default_context: "please transcribe the audio"
    - manifest_filepath: manifest2.json
        type: nemo
        weight: 1.0
        tags:
        default_context: "please translate English audio to German"
    type: group
    weight: 0.4
