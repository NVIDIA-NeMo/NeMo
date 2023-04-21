# Scripts for creation of synthetic code-switched data from monolingual sources
Follow the 2 steps listed below in order - 


1. Create the (intermediate) manifest file using `code_switching_manifest_creation.py`.  It's usage is as follows:

    `python code_switching_manifest_creation.py --manifest_language1 <absolute path of Language 1's manifest file> --manifest_language2 <absolute path of Language 2's manifest file> --manifest_save_path <absolute path to save the created manifest> --id_language1 <language code for language 1 (e.g. en)> --id_language2 <language code for language 2 (e.g. es)> --max_sample_duration_sec <maximum duration of generated sample in seconds> --min_sample_duration_sec <maximum duration of generated sample in seconds> --dataset_size_required_hrs <size of generated synthetic dataset required in hrs>`

    Estimated runtime for dataset_size_required_hrs=10,000 is ~2 mins

2. Create the synthetic audio data and the corresponding manifest file using `code_switching_audio_data_creation.py` It's usage is as follows: 

    `python code_switching_audio_data_creation.py --manifest_path <absolute path to intermediate CS manifest generated in step 1> --audio_save_folder_path <absolute path to directory where you want to save the synthesized audios> --manifest_save_path <absolute path to save the created manifest> --audio_normalized_amplitude <scaled normalized amplitude desired> --cs_data_sampling_rate <desired sampling rate for generated audios> --sample_beginning_pause_msec <pause to be added to the beginning of the generated sample in milli seconds> --sample_joining_pause_msec <pause to be added between segments while joining, in milli seconds> --sample_end_pause_msec <pause to be added to the end of the generated sample in milli seconds> --is_lid_manifest <boolean to create manifest in the multi-sample lid format for the text field, true by default> --workers <number of worker processes>`

    Example of the multi-sample LID format: ```[{“str”:“esta muestra ” “lang”:”es”},{“str”:“was generated synthetically”: “lang”:”en”}]```
    
    Estimated runtime for generating a 10,000 hour corpus is ~40 hrs with a single worker