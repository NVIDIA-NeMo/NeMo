# Changelog

## NVIDIA Neural Modules 2.0.0rc1

### Highlights

#### Large language models

- PEFT: QLoRA support, LoRA/QLora for Mixture-of-Experts (MoE) dense layer
- State Space Models & Hybrid Architecture support (Mamba2 and NV-Mamba2-hybrid)
- Support Nemotron, Minitron, Gemma2, Qwen, RAG
- Custom Tokenizer training in NeMo
- Update the Auto-Configurator for EP, CP and FSDP

#### Multimodal

- NeVA: Add SOTA LLM backbone support (Mixtral/LLaMA3) and suite of model parallelism support (PP/EP)
- Support Language Instructed Temporal-Localization Assistant (LITA) on top of video NeVA

#### ASR

- SpeechLM and SALM
- Adapters for Canary Customization
- Pytorch allocator in PyTorch 2.2 improves training speed up to 30% for all ASR models
- Cuda Graphs for Transducer Inference
- Replaced webdataset with Lhotse - gives up to 2x speedup
- Transcription Improvements - Speedup and QoL Changes
- ASR Prompt Formatter for multimodal Canary

#### Export & Deploy

- In framework PyTriton deployment with backends: - PyTorch - vLLM - TRT-LLM update to 0.10
- TRT-LLM C++ runtime

### Detailed Changelogs

#### ASR

<details><summary>Changelog</summary>

- Support dataloader as input to `audio` for transcription by @titu1994 :: PR: #9201  
- Clean up dev docs collection section by @yaoyu-33 :: PR: #9205  
- Fix Online_Offline_Microphone_VAD_Demo.ipynb by @stevehuang52 :: PR: #9251  
- Remove .nemo instead of renaming by @mikolajblaz :: PR: #9281  
- Fix GreedyBatchedCTCInfer regression from GreedyCTCInfer. by @galv :: PR: #9347
- Revert "Fix GreedyBatchedCTCInfer regression from GreedyCTCInfer." by @titu1994 :: PR: #9351
- Prompt formatter API and canary transcribe tensor input support by @pzelasko :: PR: #9206
- Fix prompt formatter's defaults=None case in multi-task model by @pzelasko :: PR: #9366
- move AED chunked infer script by @stevehuang52 :: PR: #9367
- Use model-cast-to-bfloat16 rather than AMP-to-bfloat16 for inference. by @galv :: PR: #9198
- ci: Fix `L2_Segmentation_Tool_Parallel_ctc_segmentation_test_L2_Eng_C… by @ko3n1g :: PR: #9399
- Fix logging message for ASR by @titu1994 :: PR: #9469
- Add support to change Multi task model prompt by @titu1994 :: PR: #9542
- Enable encoder adapters for Canary and MultiTaskAED models by @titu1994 :: PR: #9409
- Audio model collection by @anteju :: PR: #9263
- TitaNet Batch Verify Speaker by @monica-sekoyan :: PR: #9337
- Fix the arguments  of forward_for_export function in msdd_models by @tango4j :: PR: #9624
- chore: Pin branch in notebooks by @ko3n1g :: PR: #9697
- refactor: notebook branch release by @ko3n1g :: PR: #9711
- Canary Adapters tutorial (#9670) by @nithinraok :: PR: #9777
- typos and branch name update to r2.0.0rc1 by @nithinraok :: PR: #9846
- Fix RNNT alignments test by @artbataev :: PR: #9770
- By default trust remote code from HF Datasets by @nithinraok :: PR: #9886
- Temporarily disable cuda graph based RNN-T greedy inference for r2.0.0rc1 by @galv :: PR: #9904
- Enable CUDA graphs by default, but require CUDA 12.6 for full graphs by @artbataev :: PR: #9919
- update branch name for script by @nithinraok :: PR: #9936
- updte branch by @nithinraok :: PR: #9942
</details>

#### TTS

<details><summary>Changelog</summary>
- Clean up dev docs collection section by @yaoyu-33 :: PR: #9205
- Add mel codec checkpoints by @anteju :: PR: #9228
- GPU unit tests: Mark flaky tests to be fixed by @pablo-garay :: PR: #9559
- chore: Pin branch in notebooks by @ko3n1g :: PR: #9697
- Create __init__.py by @stevehuang52 :: PR: #9892
- [NeMo-UX] Fixes to make PreemptionCallback work by @hemildesai :: PR: #9830
- Fix Docker build. Make Dockerfile consistent with CI by @artbataev :: PR: #9784
- Multimodal data prep notebook fix by @cuichenx :: PR: #9910
- [NeMo-UX] Add distributed checkpointing unit tests by @ashors1 :: PR: #9794
- r2.0.0rc1 fix for dist checkpoint loading by @yaoyu-33 :: PR: #9854
- [NeMo-UX] Rename sdk references to NeMo Run by @hemildesai :: PR: #9872
- [NeMo-UX] Fix some serialization bugs by @ashors1 :: PR: #9868
- add mixtral neva tutorial (moe + token fusion + siglip) by @paul-gibbons :: PR: #9926
- [NeMo-UX] Add more NeMo Logger tests by @ashors1 :: PR: #9795
- Akoumparouli/mixtral fixes for r2.0.0rc1 by @akoumpa :: PR: #9911
- R2.0.0rc1 clip fix by @Slyne :: PR: #9871
- [NeMo-UX] Add missing docstrings and update some defaults by @ashors1 :: PR: #9895
- Add REST service requirements.txt by @oyilmaz-nvidia :: PR: #9923
- add bert latest fix by @JRD971000 :: PR: #9921
- remove empy reconfigure_limit_batches by @akoumpa :: PR: #9934
- fix mem by @terrykong :: PR: #9964
- Run a sample query for a quantized model conditionally by @janekl :: PR: #9965
- Add pydantic-settings  by @oyilmaz-nvidia :: PR: #9961
- Resiliency features update by @jbieniusiewi :: PR: #9714
- [NeMo-UX] Wrap task config save in a try/except by @ashors1 :: PR: #9956
- [NeMo-UX] Update default PTL logging `save_dir` by @ashors1 :: PR: #9954
- Fix lita tutorial by @Slyne :: PR: #9980
- Add deploy and REST API support to NeMo 2.0 by @athitten :: PR: #9834
</details>

## NVIDIA Neural Modules 2.0.0rc0

### Highlights

#### LLM and MM

##### Models

- Megatron Core RETRO
  - Pre-training
  - Zero-shot Evaluation

- Pretraining, conversion, evaluation, SFT, and PEFT for:
  - Mixtral 8X22B
  - Llama 3
  - SpaceGemma

- Embedding Models Fine Tuning
  - Mistral
  - BERT

- BERT models
  - Context Parallel
  - Distributed checkpoint

- Video capabilities with NeVa

##### Performance

- Distributed Checkpointing
  - Torch native backend
  - Parallel read/write
  - Async write

- Multimodal LLM (LLAVA/NeVA)
  - Pipeline Parallelism support
  - Sequence packing support

##### Export

- Integration of Export & Deploy Modules into NeMo Framework container
  - Upgrade to TRT-LLM 0.9

#### Speech (ASR & TTS)

##### Models

- AED Multi Task Models (Canary) - Multi-Task Multi-Lingual Speech Recognition / Speech Translation model
- Multimodal Domain - Speech LLM supporting SALM Model
- Parakeet-tdt_ctc-1.1b Model - RTFx of > 1500 (can transcribe 1500 seconds of audio in 1 second)
- Audio Codec 16kHz Small - NeMo Neural Audio Codec for discretizing speech for use in LLMs
  - mel_codec_22khz_medium
  - mel_codec_44khz_medium

##### Perf Improvements

- Transcribe() upgrade - Enables one line transcribe with files, tensors, data loaders
- Frame looping algorithm for RNNT faster decoding - Improves Real Time Factor (RTF) by 2-3x
- Cuda Graphs + Label-Looping algorithm for RNN-T and TDT Decoding - Transducer Greedy decoding at over 1500x RTFx, on par with CTC Non-Autoregressive models
- Semi Sorted Batching support - External User contribution that speeds up training by 15-30%.

##### Customization

- Context biasing for CTC word stamping - Improve accuracy for custom vocabulary and pronunciation
  - Longform Inference
  - Longform inference support for AED models
- Transcription of multi-channel audio for AED models

##### Misc

- Upgraded webdataset - Speech and LLM / Multimodal unified container

### Detailed Changelogs

#### ASR
  
<details><summary>Changelog</summary>

- Enable using hybrid asr models in CTC Segmentation tool by @erastorgueva-nv :: PR: #8828
- TDT confidence fix by @GNroy :: PR: #8982
- Fix union type annotations for autodoc+mock-import rendering by @pzelasko :: PR: #8956
- NeMo dev doc restructure by @yaoyu-33 :: PR: #8896
- Improved random seed configuration for Lhotse dataloaders with docs by @pzelasko :: PR: #9001
- Fix #8948, allow preprocessor to be stream captured to a cuda graph when doing per_feature normalization by @galv :: PR: #8964
- [ASR] Support for transcription of multi-channel audio for AED models by @anteju :: PR: #9007
- Add ASR latest news by @titu1994 :: PR: #9073
- Fix docs errors and most warnings by @erastorgueva-nv :: PR: #9006
- PyTorch CUDA allocator optimization for dynamic batch shape dataloading in ASR by @pzelasko :: PR: #9061
- RNN-T and TDT inference: use CUDA graphs by default by @artbataev :: PR: #8972
- Fix #8891 by supported GPU-side batched CTC Greedy Decoding by @galv :: PR: #9100
- Update branch for notebooks and ci in release by @ericharper :: PR: #9189
- Enable CUDA graphs by default only for transcription by @artbataev :: PR: #9196
- rename paths2audiofiles to audio by @nithinraok :: PR: #9209
- Fix ASR_Context_Biasing.ipynb contains FileNotFoundError by @andrusenkoau :: PR: #9233
- Cherrypick: Support dataloader as input to `audio` for transcription (#9201) by @titu1994 :: PR: #9235
- Update Online_Offline_Microphone_VAD_Demo.ipynb by @stevehuang52 :: PR: #9252
- Dgalvez/fix greedy batch strategy name r2.0.0rc0 by @galv :: PR: #9243
- Accept None as an argument to decoder_lengths in GreedyBatchedCTCInfer::forward by @galv :: PR: #9246
- Fix loading github raw images on notebook by @nithinraok :: PR: #9282
- typos by @nithinraok :: PR: #9314
- Re-enable cuda graphs in training modes. by @galv :: PR: #9338
- add large model stable training fix and contrastive loss update for variable seq by @nithinraok :: PR: #9259
- Fix conv1d package in r2.0.0rc0  by @pablo-garay :: PR: #9369
- Fix GreedyBatchedCTCInfer regression from GreedyCTCInfer. (#9347) by @titu1994 :: PR: #9350
- Make a backward compatibility for old MSDD configs in label models by @tango4j :: PR: #9377
- Force diarizer to use CUDA if cuda is available and if device=None. by @tango4j :: PR: #9380

</details>
  
#### TTS

<details><summary>Changelog</summary>

- [TTS] Add tutorial for training audio codecs by @rlangman :: PR: #8723
- Update radtts.py by @blisc :: PR: #9097
- [Nemo CICD] RADTTS test optional by @pablo-garay :: PR: #9112
- Remove Radtts CI test by @blisc :: PR: #9144
- Fix T5 G2P Input and Output Types by @blisc :: PR: #9224

</details>

#### LLM and MM

<details><summary>Changelog</summary>

- Rachitg/dpa by @rachitgarg91 :: PR: #8911
- Remove precision args in trainer due to PTL update by @yaoyu-33 :: PR: #8908
- Huvu/mcore retro by @huvunvidia :: PR: #8861
- fsdp tp > 1 bug fix by @dimapihtar :: PR: #8947
- Fix memory leak at loss func by @minitu :: PR: #8868
- change the condition for get qkv tensor from linear_qkv output in mcoremixin by @HuiyingLi :: PR: #8965
- Add safety checks for 'data' key in MegatronGPTModel cfg by @HuiyingLi :: PR: #8991
- [NeMo-UX] Adding MegatronParallel by @cuichenx :: PR: #8987
- Skip top_p computations when set to 1.0 by @odelalleau :: PR: #8905
- Gemma bug by @cuichenx :: PR: #8962
- [NeMo-UX] Adding megatron strategy by @marcromeyn :: PR: #8995
- Quantized checkpoint support in export and deploy modules by @janekl :: PR: #8859
- add geglu to mlp swap by @JRD971000 :: PR: #8999
- add timeout for new_group by @acphile :: PR: #8998
- Zero-shot evaluation pipeline for mcore RETRO by @huvunvidia :: PR: #8941
- Added fusion for squared relu by @sanandaraj5597 :: PR: #8963
- Developer Documents for mcore RETRO by @huvunvidia :: PR: #9026
- [NeMo-UX] Adding GPTModel & MockDataModule by @marcromeyn :: PR: #9011
- Adding unit test for mcore RETRO model by @huvunvidia :: PR: #9022
- docs and simplification of cmd args by @arendu :: PR: #8979
- [NeMo-UX] Add checkpoint-io to MegatronStrategy by @marcromeyn :: PR: #9057
- Enable Sequence Packing and Pipeline Parallel in NeVA by @yaoyu-33 :: PR: #8957
- Mingyuanm/add back fp8 support to sd by @Victor49152 :: PR: #9070
- unfused lora by @arendu :: PR: #9004
- Handle case where num_query_groups is set to null for LoRA config setup by @vysarge :: PR: #9075
- Alit/griffin by @JRD971000 :: PR: #9021
- Implement DistributedCheckpointIO by @mikolajblaz :: PR: #9016
- Video Neva Pretraining + Inference Implementation by @paul-gibbons :: PR: #9095
- HF to .nemo for Mixtral-8x22B-instruct by @akoumpa :: PR: #9060
- mcore ds updates by @dimapihtar :: PR: #8951
- Alit/griffin perf by @JRD971000 :: PR: #9107
- Add assert for max_steps to be positive in MegatronGPTSFTModel by @athitten :: PR: #9110
- Extend sequence length padding for GPT SFT to account for context parallel by @vysarge :: PR: #8869
- Update gpt dataset config parameter for mock by @thomasdhc :: PR: #9118
- Add Mcore DistributedDataParallel and distributed optimizer into Nemo by @gdengk :: PR: #9034
- Revert "Add assert for max_steps to be positive in MegatronGPTSFTMode… by @pablo-garay :: PR: #9128
- scripts to convert HF lora to nemo by @arendu :: PR: #9102
- Prevent duplicated checkpoints by @mikolajblaz :: PR: #9015
- add TN/ITN link in speech tools list by @erastorgueva-nv :: PR: #9142
- Cleanup deprecated files and temporary changes by @cuichenx :: PR: #9088
- Use DP+CP groups as the FSDP sharding domain by @erhoo82 :: PR: #9145
- CUDA memory profile by @erhoo82 :: PR: #9096
- Fix missing func for T5 model by @gdengk :: PR: #9141
- Add knob for load_directly_on_device by @mikolajblaz :: PR: #9125
- Revert rope fusion defaults by @cuichenx :: PR: #9238
- Update nemo.export module for quantized models by @janekl :: PR: #9250
- Fix circular import for MM dataprep notebook by @cuichenx :: PR: #9287
- neva media_type + text generation default fix by @paul-gibbons :: PR: #9257
- fix lora and ptuning and isort/black by @oyilmaz-nvidia :: PR: #9290
- add check if num layers is divisible by pp size by @dimapihtar :: PR: #9208
- Fix P-tuning for Llama based models by @apanteleev :: PR: #9297
- add deprecation warnings by @pablo-garay :: PR: #9266
- move pooler under post_process by @dimapihtar :: PR: #9328
- add deprecation note for nmt by @dimapihtar :: PR: #9342
- Fix incorrect checkpoint removal logic (#9192) by @mikolajblaz :: PR: #9204
- fix fp16 precision issue by @dimapihtar :: PR: #9376
- Fix module.training for Neva in FusedAttn backward which causes nan by @yaoyu-33 :: PR: #8877

</details>

#### Export

<details><summary>Changelog</summary>

- Updates for TRT-LLM 0.9 by @oyilmaz-nvidia :: PR: #8873
- Mingyuanm/sdxl export by @Victor49152 :: PR: #8926
- Avoid unpacking NeMo checkpoints before exporting to TRT-LLM by @apanteleev :: PR: #8866
- Update gemma for trt-llm 0.9 by @oyilmaz-nvidia :: PR: #8974
- TRT-LLM export P-tuning related fixes by @apanteleev :: PR: #8863

</details>

#### General Improvements

<details><summary>Changelog</summary>

- Update package info by @ericharper :: PR: #8793
- [Nemo CICD] Update mcore 4.13.24 by @pablo-garay :: PR: #8917
- Akoumparouli/low mem mixtral ckpt converter by @akoumpa :: PR: #8895
- Adding RETRO tests to Action Tests (cicd-main.yml)  by @huvunvidia :: PR: #8942
- Akoumparouli/fix sd train 2 by @akoumpa :: PR: #8883
- Update te install for jenkins by @ericharper :: PR: #8954
- [Nemo CICD] Add last job depending on others for blocking check by @pablo-garay :: PR: #8959
- Minor quantization pipeline updates by @janekl :: PR: #8924
- Fix External CLIP Converter by @yaoyu-33 :: PR: #8960
- PP support in LoRA merge script by @cuichenx :: PR: #8934
- Update PR template by @ericharper :: PR: #8978
- Update Latest News by @shashank3959 :: PR: #8837
- Fix incorrect link to latest news in README by @shashank3959 :: PR: #8985
- Update dependency install for LLM and MM by @ericharper :: PR: #8990
- Temporarily remove mcore dep by @ericharper :: PR: #9010
- [Nemo CICD] further specialize runners for more parallelism by @pablo-garay :: PR: #9036
- Update mm dataprep notebook based on feedback by @cuichenx :: PR: #9029
- Fix import in lora merge script by @cuichenx :: PR: #9032
- [Nemo CICD] Run when labeled:Run CICD by @pablo-garay :: PR: #9044
- [Nemo CICD] Add tag/label for 1-gpu runner by @pablo-garay :: PR: #9046
- [Nemo CICD] checkout v4 by @pablo-garay :: PR: #9048
- [Nemo CICD] Remove temp test change by @pablo-garay :: PR: #9049
- remove in-place addition for dreambooth train with text encoder by @Victor49152 :: PR: #8825
- Mingyuanm/sdxl quantization notebook by @Victor49152 :: PR: #9042
- [Nemo CICD] Trigger on comment issued by @pablo-garay :: PR: #9062
- zarr ckpt to torch_dist ckpt converter by @dimapihtar :: PR: #8842
- Restore PTQ tests for Llama2 (reopened) by @janekl :: PR: #9064
- add clip H config by @JRD971000 :: PR: #9082
- [NeMo-UX] Add mixed-precision plugin by @marcromeyn :: PR: #9065
- Comment baichuan test and update pr template by @ericharper :: PR: #9085
- Add safe extraction of nemo tar files by @athitten :: PR: #8976
- Improved `shard_id` parsing in `LazyNemoTarredIterator`, enables AIS dataloading by @pzelasko :: PR: #9077
- [NeMo-UX] Add mistral-7b model by @marcromeyn :: PR: #9066
- Llama3 Conversion Script Update by @suiyoubi :: PR: #9089
- dehardcode test string by @JimmyZhang12 :: PR: #8865
- [Nemo CICD] Try trigger cicd run on comment by @pablo-garay :: PR: #9111
- Lhotse dataloading: RIR augmentation and nemo/tarred input support for RIR and noise aug by @pzelasko :: PR: #9109
- mixtral evaluation PR by @Slyne :: PR: #8989
- [Nemo CICD] Revert: run GHA cicd on comment by @pablo-garay :: PR: #9119
- [Nemo CICD] Comment out flaky test: running too long by @pablo-garay :: PR: #9123
- [Nemo CICD] Add timeout to unit tests by @pablo-garay :: PR: #9132
- [Nemo CICD] Indicate optional test in name (prefix) by @pablo-garay :: PR: #9139
- video neva null image+video folder path fix by @paul-gibbons :: PR: #9116
- [NeMo-UX] Add data module by @cuichenx :: PR: #9133
- NeMo Inference Requirements by @oyilmaz-nvidia :: PR: #9093
- Remove debug print by @maanug-nv :: PR: #9074
- Remove legacy CI by @pablo-garay :: PR: #9149
- Update support for push_to_hf_hub() by @titu1994 :: PR: #9159
- [Nemo CICD] comment out flaky PTQ tests by @pablo-garay :: PR: #9160
- Update branch by @ericharper :: PR: #9211
- dist adam transpose fix by @dimapihtar :: PR: #9239
- [Nemo CICD] Increase time limit for Speech_Checkpoints_tests (#9186) by @pablo-garay :: PR: #9247
- Pin transformers by @ericharper :: PR: #9261
- Fix typo in HF tutorial by @titu1994 :: PR: #9302

</details>

## NVIDIA Neural Modules 1.23.0

### Highlights

#### Models

##### Nvidia Starcoder 2 - 15B

- Announcement - https://developer.nvidia.com/blog/unlock-your-llm-coding-potential-with-starcoder2/
- AI Foundation Model Inference  - https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-foundation/models/starcoder2-15b
- https://huggingface.co/bigcode/starcoder2-15b

##### NeMo Canary
Announcement - https://nvidia.github.io/NeMo/blogs/2024/2024-02-canary/

- https://huggingface.co/nvidia/canary-1b

#### NeMo LLM

- Falcon
- Code Llama
- StarCoder
- GPT perf improvements
- Context parallelism
- Mistral
- Mixtral (without expert parallelism)
- Mcore GPT Dataset integration

#### NeMo MM
- CLIP
- Stable Diffusion (supporting LoRA)
- Imagen
- ControlNet (for SD)
- Instruct pix2pix (for SD)
- LLAVA
- NeVA
- DreamFusion++
- NSFW filtering

#### NeMo ASR

- Lhotse Dataloading support #7880
- Canary: Multi task multi lingual ASR #8242
- LongForm Audio for Diarization #7737
- Faster algorithm for RNN-T Greedy #7926
- Cache-Aware streaming notebook #8296

#### NeMo TTS

#### NeMo Vision

#### Known Issues

##### ASR

###### RNNT WER calculation when fused batch size > 1 during validation / test step()

Previously, the RNNT metric was stateful while the CTC one was not ([r1.22.0](https://github.com/NVIDIA/NeMo/blob/r1.22.0/nemo/collections/asr/metrics/rnnt_wer_bpe.py#L419-L420), [r1.23.0](https://github.com/NVIDIA/NeMo/blob/r1.23.0/nemo/collections/asr/metrics/wer.py#L333))

Therefore this calculation in the RNNT joint for fused operation worked properly. However with the unification of metrics in r1.23.0, a bug was introduced where only the last sub-batch of metrics calculates the scores and does not accumulate. This is patched via https://github.com/NVIDIA/NeMo/pull/8587 and will be fixed in the next release.

**Workaround**: Explicitly disable fused batch size during inference using the following command 

```python
from omegaconf import open_dict
model = ...
decoding_cfg = model.cfg.decoding
with open_dict(decoding_cfg):
  decoding_cfg.fused_batch_size = -1
model.change_decoding_strategy(decoding_cfg)
```

Note: This bug does not affect scores calculated via model.transcribe() (since it does not calculate metrics during inference, just text), or using the `transcribe_speech.py` or `speech_to_text_eval.py` in `examples/asr`.

###### Two failing unit tests due to a change in expected results, caused by lhotse version update

#### Container

For additional information regarding NeMo containers, please visit: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo

`docker pull nvcr.io/nvidia/nemo:24.01.speech`

#### ASR

<details><summary>Changelog</summary>

- Update link to yaml file in ASR_with_Transducers.ipynb by @Faith-Nchifor :: PR: #8014
- Use convert_hf_dataset_to_nemo by @karpnv :: PR: #8017
- Update asr_language_modeling.rst: Add a missing word by @martin0258 :: PR: #8007
- spelling mistake by @orena1 :: PR: #7903
- update asr eval by @stevehuang52 :: PR: #8045
- fix noise aug by @stevehuang52 :: PR: #8057
- Various fixes for typos and urls by @titu1994 :: PR: #8066
- [Fix] Increase length check tolerance to prevent test failing by @anteju :: PR: #8067
- Add text metrics to asr eval by @stevehuang52 :: PR: #8087
- fix device setting to allow using accelerator cpu by @orena1 :: PR: #8084
- .ctm in data simulator annotator compliant with RT-09 specification by @popcornell :: PR: #8004
- Fix AST eval by @stevehuang52 :: PR: #8112
- fix: numba.*_num_threads resets torch num_threads #8141 by @itzsimpl :: PR: #8145
- Update dependencies by @titu1994 :: PR: #8156
- NeMo + Lhotse integration by @pzelasko :: PR: #7880
- Speedup RNN-T greedy decoding by @artbataev :: PR: #7926
- [docker] Install k2 before NeMo for faster image rebuilding by @pzelasko :: PR: #8204
- [docs] Add --force_codec to tarred dataset creation examples by @pzelasko :: PR: #8227
- Temporarily use the previous RNN-T decoding algorithm as default by @artbataev :: PR: #8226
- Make TDT inference not require duration params by @hainan-xv :: PR: #8207
- Cache Aware Streaming tutorial notebook by @erastorgueva-nv :: PR: #8296
- fix path location and branch by @nithinraok :: PR: #8304
- Attention encoder-decoder models for multiple speech-to-text tasks  … by @titu1994 :: PR: #8324
- Remove asr webapp by @titu1994 :: PR: #8347
- remove _target_ at model level in aed model config [ASR] by @krishnacpuvvada :: PR: #8351
- Add change_vocabulary and save_tokenizers() support to Multitask ASR models by @titu1994 :: PR: #8357
- Change default beam size by @titu1994 :: PR: #8371
-  adding jenkins test for speech_to_text_aed model by @krishnacpuvvada :: PR: #8368
- Add Finetuning tutorial with HF Datasets by @nithinraok :: PR: #8356
- wer fix by @tbartley94 :: PR: #8404
- add ensemble decoding fix by @nithinraok :: PR: #8427
- Update k2 by @artbataev :: PR: #8492

</details>

#### TTS

<details><summary>Changelog</summary>

- [TTS] Scale sampler steps by number of devices by @rlangman :: PR: #7947
- Add All Multimodal Source Code Part 2: Text to image, x to nerf by @yaoyu-33 :: PR: #7970
- [TTS] Add period discriminator and feature matching loss to codec recipe by @rlangman :: PR: #7884
- Added VectorQuantizer base class by @anteju :: PR: #8011

</details>

#### LLMS

<details><summary>Changelog</summary>

- Add interface to set NCCL options of each process group by @erhoo82 :: PR: #7923
- Support O2 training of PEFT and SFT by @cuichenx :: PR: #7971
- [NLP] Access scaler only in FP16 case by @janekl :: PR: #7916
- [NLP] Minor improvements in Llama conversion script by @janekl :: PR: #7978
- [NLP] Use helpers from utils_funcs.py in Llama conversion by @janekl :: PR: #7979
- [NLP] Remove replace_sampler_ddp (deprecated in Trainer) by @janekl :: PR: #7981
- Reworked MegatronPretrainingRandomBatchSampler to correctly handle epochs > 1 by @trias702 :: PR: #7920
- Remove deprecated arguments from TE's TransformerLayer by @jbaczek :: PR: #7917
- Add All Multimodal Source Code by @yaoyu-33 :: PR: #7791
- First draft of mcore bert model in NeMo by @shanmugamr1992 :: PR: #7814
- Support Falcon Variants (7B/40B/180B) in Mcore NeMo by @xuanzic :: PR: #7666
- FSDP + Tensor Parallelism by @erhoo82 :: PR: #7897
- Packed Sequence by @cuichenx :: PR: #7945
- Adding method back that was removed accidentally by @ericharper :: PR: #8038
- [NLP] ArtifactItem with init=True to make it debuggable by @janekl :: PR: #7980
- SFT patch: (1) enable sequence parallelism and (2) enable profile by @erhoo82 :: PR: #7963
- migration to PTL 2.0 for spellmapper model by @bene-ges :: PR: #7924
- Change the megatron config lr scheduler default and fix to change partitions script by @shan18 :: PR: #8094
- (1) Add SHARP interface to M-CORE, (2) use send/recv to send train loss to the first rank instead of b-cast by @erhoo82 :: PR: #7793
- Reconfigure limit_val_batches only for int by @athitten :: PR: #8099
- Fixing wrapper and moving it to base class by @shanmugamr1992 :: PR: #8055
- fix gated_linear_unit bug by @Agoniii :: PR: #8042
- Fix Adapter for MCore models by @cuichenx :: PR: #8124
- add war fix for sync issues by @gshennvm :: PR: #8130
- Improve PEFT UX by @cuichenx :: PR: #8131
- Enhance flexibility by passing callbacks as method argument by @michal2409 :: PR: #8015
- context parallelism by @xrennvidia :: PR: #7739
- Make pipelined TP comm overlap available with mcore by @erhoo82 :: PR: #8005
- remove deprecated scripts by @arendu :: PR: #8138
- adding OnlineSampleMapping by @arendu :: PR: #8137
- Add distopt support for FP8 params and BF16 optimizer state by @timmoon10 :: PR: #7909
- Revert adding OnlineSampleMapping by @pablo-garay :: PR: #8164
- Token count and sequence length logging for MegatronGPTSFTModel by @vysarge :: PR: #8136
- Use latest apex internal API by @jbaczek :: PR: #8129
- tune specific params in the base model by @arendu :: PR: #7745
- Virtual pipeline parallel support for MegatronGPTSFTModel by @vysarge :: PR: #7964
- removed deprecated peft model by @arendu :: PR: #8183
- remove more deprecated files by @arendu :: PR: #8169
- Pre-generate cu_seqlens argmin and max_seqlen to remove host-to-device sync by @erhoo82 :: PR: #8108
- Add the interface to use SHARP to FSDP strategy by @erhoo82 :: PR: #8202
- Multimodal required NLP base model changes by @yaoyu-33 :: PR: #8188
- [NLP] Improve and unify loading state_dict for community models by @janekl :: PR: #7977
- Rename Finetuning Scripts by @cuichenx :: PR: #8201
- Final multimodal PR with our recent developments on MM side by @yaoyu-33 :: PR: #8127
- Add include_text parameter to SFT dataloaders by @Kipok :: PR: #8198
- Add random_seed argument to generate by @Kipok :: PR: #8162
- Added support for neptune logger by @harishankar-gopalan :: PR: #8210
- Pre-compute max_seqlen and cu_seqlens_argmin in all model-parallel cases by @erhoo82 :: PR: #8222
- Use PackedSeqParams in accordance with changes in Megatron-LM by @cuichenx :: PR: #8205
- Fix to peft & virtual pipeline parallel unsupported check by @vysarge :: PR: #8216
- Fixed the tp overlap switch by @sanandaraj5597 :: PR: #8195
- add knobs for rope/swiglu fusion by @lhb8125 :: PR: #8184
- Added sample cpu_offloading switch to YAML by @sanandaraj5597 :: PR: #8148
- Syncing random seed between ranks in generate by @Kipok :: PR: #8230
- add first_val_step to mcore scheduler by @JimmyZhang12 :: PR: #8150
- Correct padding for SFT input data to account for sequence parallel + TE's fp8 op dimension requirements by @vysarge :: PR: #8240
- Mistral 7b conversion script by @akoumpa :: PR: #8052
- switch to mcore dataset [with FIM support] by @dimapihtar :: PR: #8149
- Mixtral to NeMo conversion script. by @akoumpa :: PR: #8155
- fixes to accomendate mcore changes by @HuiyingLi :: PR: #8261
- Allow MegatronPretrainingRandomSampler to do multi-epoch training by @trias702 :: PR: #8239
- Add dist ckpt support for regular optimizers by @mikolajblaz :: PR: #7749
- add deallocate pipeline output optimization by @JimmyZhang12 :: PR: #8279
- Fix memory leak caused by context parallelism hanging references by omegaconf by @JimmyZhang12 :: PR: #8299
- distributed fused adam + rampup bs support by @dimapihtar :: PR: #8302
- Update PEFT Doc by @cuichenx :: PR: #8262
- Converter script fixes for mixtral/mistral by @akoumpa :: PR: #8272
- Keep max_seqlen and cu_seqlens_argmin for later micro-batches when PP>1 by @erhoo82 :: PR: #8334
- Enable megatron core loggers for GPT pretraining by @ashbhandare :: PR: #8354
- mcore ds fix by @dimapihtar :: PR: #8283
- release updates by @dimapihtar :: PR: #8378
- Mcore customization doc by @HuiyingLi :: PR: #8298
- updated link to pubmed by @nithinraok :: PR: #8402
- mcore customization doc minor fix by @HuiyingLi :: PR: #8421
- Fixing mcore bert for TP, PP and SP by @shanmugamr1992 :: PR: #8336
- Add settings to suppress bf16 compile errors in CI on V100 by @athitten :: PR: #8481
- MoE parameter passing by @akoumpa :: PR: #8255
- Add fp8 support for SD/Update notebook paths by @Victor49152 :: PR: #8489

</details>

#### NeMo Tools

<details><summary>Changelog</summary>

- SDE bugfix log by @Jorjeous :: PR: #8430

</details>

#### General Improvements

<details><summary>Changelog</summary>

- Add news section to README by @ericharper :: PR: #7984
- Fixing conversion script to work for code llama by @shanmugamr1992 :: PR: #7997
- Fix crash when converting to mcore a model using rotary embeddings by @odelalleau :: PR: #7998
- Added a procedure for Windows users, README by @Jorjeous :: PR: #7942
- Update manifest.py to speedup loading tarred datasets by @stevehuang52 :: PR: #7900
- [Fix] Fixed name of a test by @anteju :: PR: #7986
- Fix lora merge script by @cuichenx :: PR: #8113
- Support transcoding audio formats when saving tarred datasets (FLAC, OPUS) by @pzelasko :: PR: #8102
- README edit to change Apple Silicon install instructions (to fix a break introduced by pytorch 2) by @stephenmcconnachie :: PR: #8122
- Fixes NVIDIA/apex installation to not erroneously install the  pkg by @terrykong :: PR: #8126
- Graphviz fix by @GNroy :: PR: #7843
- Update README.rst by @fayejf :: PR: #8154
- Fix TP>1 issue for conversion script by @cuichenx :: PR: #8144
- Support torch jit script by @artbataev :: PR: #8027
- NeMo Multimodal Docs and Tests Initial PR by @yaoyu-33 :: PR: #8028
- Remove left-over prints in NeMo+Lhotse code by @pzelasko :: PR: #8180
- Upgrade to DLFW PyTorch 23.12 by @ericharper :: PR: #8163
- Add Lhotse support for  key in NeMo manifests by @pzelasko :: PR: #8197
- Fix CPU Initialization and TP>1 for LoRA Merge Script by @cuichenx :: PR: #8199
- Add support in Neural Typecheck to disable semantic checks by @titu1994 :: PR: #8212
- Pin lhotse=1.19.2 in r1.23.0 by @pzelasko :: PR: #8303
- Multimodal r1.23.0 bug fix  by @yaoyu-33 :: PR: #8315
- MCore dataset compatibility for tokenizers by @vysarge :: PR: #8390
- Update NFA video download link by @erastorgueva-nv :: PR: #8406
- Update MM Dataprep Tutorial by @cuichenx :: PR: #8410
- Fix dreambooth data sampler issue by @yaoyu-33 :: PR: #8400
- Fix a bug in CTM line processing function for multi-speaker data simulations by @tango4j :: PR: #8416
- Akoumparouli/mistral bugfix by @akoumpa :: PR: #8353
- pin to 0.5.0 by @ericharper :: PR: #8465
- Update NeMo Multimodal Requirements by @yaoyu-33 :: PR: #8515
- Fix link in multimodal dataprep tutorial by @cuichenx :: PR: #8517

</details>