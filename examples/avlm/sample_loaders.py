import os
import re
import json
import torch

from collections import defaultdict

import inflect
plural = inflect.engine()

from nemo.collections.multimodal.data.energon import (
    ImageToken,
    AudioToken,
    SoundToken,
    SpeechToken,
    VideoToken,
)
from nemo.collections.avlm.data.energon import AVLMMediaDict


def get_media(raw, media_type, value, offset=None, duration=None):
    """
    Return:
        if media_type == 'text', return the text string
        if media_type == 'image', return as PIL Image
        if media_type == 'audio' or 'video', return as AVLMMediaDict
    """
    assert media_type in ["text", "audio", "video", "image"]

    if media_type == "text":
        return value
    else:
        media_dict = { "media_type": media_type, "media_value": raw[value]}
        if offset is not None:
            media_dict["offset"] = offset
        if duration is not None:
            media_dict["duration"] = duration
        return AVLMMediaDict(**media_dict)


"""
(interleaved-sample-loader)=
## Example: Interleaved Data and Arbitrary Media Count

### The Webdataset Structure

If you need multiple files with an arbitrary number of data per sample, e.g. multiple image / video / audio files, this shows a blueprint for how to setup your webdataset tar files and how to load that webdataset with Energon.

The structure of the shard files could be like this:

`tar -tvf shard_0.tar`:
```python
sample_000001.2345ew.jpg
sample_000001.json
sample_000002.35tags.mp4
sample_000002.as23ds.jpg
sample_000002.gd1dtg.wav
sample_000002.gds233.jpg
sample_000002.json
sample_000002.sdag42.jpg
sample_000003.json
sample_000004.asf234.wav
sample_000004.json
```

where the structure of a json file is:

`tar -xf shard_0.tar sample_000001.json -O`:
```json
{
    "audios": [null, null, null],
    "videos": [null, null, null],
    "duration": [null, null, null],
    "offset": [null, null, null],
    "images": [null, "2345ew.jpg", null],
    "texts": ["This is some text, an image is following.", null, "More text after the image."],
}
```
Note that the image path corresponds to the filename of the image after the first "." in the sample. This is all part of the extension as defined by webdataset. Everything before the first "." is part of the sample key and must be equal to match into the same group.
"""


def sample_loader_interleaved(raw: dict) -> dict:
    # Note that only the images are decoded, all other files are read as raw bytes.
    jsn = json.loads(raw["json"])
    sequence = []
    for text, audio, video, image, offset, duration in zip(
        jsn["texts"],
        jsn.get("audios") or [None]*len(jsn["texts"]), 
        jsn.get("videos") or [None]*len(jsn["texts"]), 
        jsn["images"],        
        jsn.get("offset") or [None]*len(jsn["texts"]), 
        jsn.get("duration") or [None]*len(jsn["texts"]), 
    ):
        media = [("text", text), ("audio", audio), ("video", video), ("image", image)]
        sequence.append(get_media(raw,t,v,offset,duration) for t, v in media if v is not None)

    return dict(__key__=raw["__key__"], 
        sequence=sequence,
    )


def part_filter_interleaved(part: str) -> bool:
    # Need to load all parts
    return True


"""
(multi-turn-sample-loader)=
## Example: Interleaved Data and Arbitrary Media Count

### The Webdataset Structure

The structure of the shard files could be like this:

`tar -tvf shard_0.tar`:
```python
sample_000001.2345ew.flac
sample_000001.35tags.mp4
sample_000001.as23ds.jpg
sample_000001.gd1dtg.wav
sample_000001.gds233.jpg
sample_000001.json
sample_000002.asf234.wav
sample_000002.json
sample_000003.json
```

```json structure 1
{
  "audios": "sample_000001.2345ew.flac,sample_000001.gd1dtg.wav", # or "audios": [sample_000001.2345ew.flac,sample_000001.gd1dtg.wav]
  "audio_durations": [5.3058125, 3.06238],
  "videos": "sample_000001.35tags.mp4", # or "videos": [sample_000001.35tags.mp4]
  "video_durations": [5.607625],
  "images": "sample_000001.as23ds.jpg,sample_000001.gds233.jpg", # or "images": [sample_000001.as23ds.jpg,sample_000001.gds233.jpg]
  "conversations": [
    {
      "from": "User",
      "value": "<audio>"
    },
    {
      "from": "Assistant",
      "value": "Automatic speech recognition is a technology that allows computers to recognize and transcribe spoken language. In the NeMo Framework, ASR is used for tasks such as speech-to-text and voice recognition."
    },
    {
      "from": "User",
      "value": "Describe what is NeMo based on the tutorial video: <video> and the information in the two images: <image> <image>. Combine that information with sound <audio>. Answer: "
    },
    {
      "from": "Assistant",
      "value": "The NeMo Framework provides a range of tools and features for training and deploying ASR models, including model parallelism, data parallelism, and distributed checkpointing. This allows for faster training and inference times, as well as improved model accuracy and reliability."
    }
  ]
}
```

```json structure 2
{
  "conversations": [
    {
      "type": "audio",
      "from": "User",
      "duration": 5.3058125,
      "value": "2345ew.flac"
    },
    {
      "type": "text",
      "from": "Assistant",
      "value": "Automatic speech recognition is a technology that allows computers to recognize and transcribe spoken language. In the NeMo Framework, ASR is used for tasks such as speech-to-text and voice recognition."
    },
    {
      "type": "text, video, text, image, text, image, text, audio, text",
      "from": "User",
      "duration": [null, 5.607625, null, null, null, null, null, 3.06238, null ],
      "value": ["Describe what is NeMo based on the tutorial video: ", 
        "35tags.mp4", 
        " and the information in the two images: ", 
        "as23ds.jpg", 
        " ",
        "gds233.jpg", 
        ". Combine that information with sound ",
        "gd1dtg.wav",
        ". Answer: ",
      ]
    },
    {
      "type": "text",
      "from": "Assistant",
      "value": "The NeMo Framework provides a range of tools and features for training and deploying ASR models, including model parallelism, data parallelism, and distributed checkpointing. This allows for faster training and inference times, as well as improved model accuracy and reliability."
    }
  ]
}
```

"""


QAMediaTokenTypeMapping = {
    AudioToken().media_type: AudioToken().token_str,
    SoundToken().media_type: SoundToken().token_str,
    SpeechToken().media_type: SpeechToken().token_str,
    VideoToken().media_type: VideoToken().token_str,
    ImageToken().media_type: ImageToken().token_str,
}

MediaKeys = [k for k in QAMediaTokenTypeMapping]

def sample_loader_QA(raw: dict) -> dict:
    # Note that all files are read as raw bytes
    jsn = json.loads(raw["json"])
    output_dict = defaultdict(list)

    # process structure 1
    for media in MediaKeys:
        media_plural = plural.plural_noun(media)
        media_key = media if media in jsn else media_plural if media_plural in jsn else None

        if media_key:
            media_files = jsn[media_key]
            if isinstance(media_files, str):
                media_files = media_files.split(",")
            assert isinstance(media_files, list)
            offsets = jsn.get(media+"_offsets") or [None] * len(media_files)
            durations = jsn.get(media+"_durations") or [None] * len(media_files)
            if not isinstance(offsets, list):
                offsets = [offsets]
            if not isinstance(durations, list):
                durations = [durations]
            output_dict[media_plural] = [get_media(raw, media, f.split('.',1)[1], offsets[i], durations[i]) 
                for i,f in enumerate(media_files)]

    for turn in jsn["conversations"]:
        if "type" not in turn:
            # process structure 1
            string = turn["value"]
        else:
            # process structure 2
            string = ""
            types = [t.strip().lower() for t in turn["type"].split(",")]
            values = turn["value"]
            if not isinstance(values, list):
                values = [values]

            offsets = turn.get("offset") or [None]*len(values)
            durations = turn.get("duration") or [None]*len(values)
            if not isinstance(offsets, list):
                offsets = [offsets]
            if not isinstance(durations, list):
                durations = [durations]

            for t, v, offset, duration in zip(types, values, offsets, durations):
                raw_media = get_media(raw, t, v, offset, duration)
                if t == "text":
                    string += raw_media
                else:
                    string += QAMediaTokenTypeMapping[t]
                    output_dict[t+'s'].append(raw_media)

        if turn["from"].lower() == "assistant" or turn["from"].lower() == "gpt":
            output_dict["answers"].append(string)
        elif turn["from"].lower() == "user" or turn["from"].lower() == "human":
            output_dict["context"].append(string)

    return dict(
        __key__=raw["__key__"],
        context=output_dict["context"],
        answers=output_dict["answers"] if output_dict["answers"] else None,
        audios=output_dict["audios"] if output_dict["audios"] else None,
        videos=output_dict["videos"] if output_dict["videos"] else None,
        images=output_dict["images"] if output_dict["images"] else None,
    )


def part_filter_QA(part: str) -> bool:
    # Need to load all parts
    return True
