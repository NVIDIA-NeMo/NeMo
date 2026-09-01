Streaming Inference
===================

The speechlm2 collection provides a streaming inference pipeline for
NemotronVoiceChat that processes audio chunk by chunk, producing text and speech
output incrementally.  The pipeline follows a similar API to the NeMo ASR Inference Pipelines
(see ``nemo.collections.asr.inference``).

There are two ways to use the pipeline:

* ``StreamingS2SPipeline.run()`` processes complete audio files.  It is used by
  ``s2s_streaming_infer.py`` for a single ``.wav`` file, a directory of ``.wav``
  files, or a manifest.
* ``StreamingS2SPipeline.generate_step()`` processes one batch of ``Frame``
  objects and returns incremental outputs for that step.  Use it for servers,
  microphone connectors, or other live audio sources.

.. code-block:: text

    File inputs: one or more .wav files
    (single path, directory, or manifest)
                    │
                    ▼
              run(audio_filepaths)
                    │  creates Frame chunks
                    ▼
              generate_step(frames)
                    │
                    ├─ incremental agent audio + text
                    ├─ incremental user ASR text (when the checkpoint has an ASR head)
                    └─ [EXPERIMENTAL] incremental function head output (when the checkpoint has a function call head, as in NVIDIA-NemotronLabs-VoiceChat-11B)

Each audio file passed to ``run()`` is treated as one continuous audio stream. ``run()``
accumulates the per-step outputs for each stream and writes final audio/text
artifacts when the stream ends.

The script can append trailing silence so the agent is more likely to finish
speaking before the stream ends.  When a manifest contains reference ``text``
fields, it also reports WER for the recognized user speech.

Streaming inference is single-stream: ``streaming.batch_size`` must be ``1``.

Script Call Path
----------------

The ``s2s_streaming_infer.py`` script follows this call path:

.. code-block:: text

    Entry Script          s2s_streaming_infer.py
         │
         ▼
    Pipeline              StreamingS2SPipeline.run()
         │                  - audio buffering
         │                  - state management
         │                  - file I/O
         ▼
    Model Wrapper         NemotronVoicechatInferenceWrapper
         │                  - infer_one_step()
         │                  - perception
         │                  - model_llm_interface    (PyTorchLLM)
         │                  - model_eartts_interface (PyTorchEarTTS)
         │                    (replaced by independent AsyncOmni engines
         │                     when a component is vllm_omni)
         │                  - codec decode
         ▼
    Model                 NemotronVoiceChat
                            - DuplexSTTModel + DuplexEARTTS

(With ``s2s.decode_audio=false``, the model still predicts text and any
checkpoint-provided auxiliary tokens, but skips EarTTS generation and codec
decoding.)

Quick Start
-----------

File-Based Inference from a Script
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Call the Python script and pass configuration values as command-line overrides:

.. code-block:: bash

    python examples/speechlm2/nemo_inference_pipelines/s2s_streaming_infer.py \
        --config-path=examples/speechlm2/nemo_inference_pipelines/conf \
        --config-name=s2s_streaming \
        audio_file=/path/to/audio_or_directory_or_manifest.json \
        output_dir=./generated \
        s2s.model_path=nvidia/NVIDIA-NemotronLabs-VoiceChat-11B \
        s2s.speaker_name=Aria \
        s2s.llm_engine_type=native \
        s2s.tts_engine_type=native \
        s2s.system_prompt="You are a helpful assistant." \
        streaming.chunk_size_in_secs=0.24 \
        streaming.buffer_size_in_secs=1.68

This will:

1. Load the NemotronVoiceChat checkpoint.
2. Stream each audio file through the pipeline in chunks.
3. Save per-stream output files under ``output_dir``: generated ``.wav``,
   stereo input+output ``.wav``, ``.txt``, and per-token ``.ctm``.
4. Write ``output_processed.json`` and ``output_raw.json`` summarising the run.

Public checkpoint
^^^^^^^^^^^^^^^^^

The public weights are
`NVIDIA-NemotronLabs-VoiceChat-11B
<https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-11B>`_.
Pass the Hugging Face repository ID as ``s2s.model_path`` (shown in Quick Start
above). The first run downloads into the Hugging Face cache; a local directory
works for offline use. The registered speaker name is ``Aria``.

That checkpoint has a function-token channel and no duplex ASR head. The
pipeline feeds the previous function token back into the next frame and
exposes a decoded copy on the function output; it does not execute tool
calls. User-transcription fields are empty and ASR-based forced turn-taking
is disabled; the model's learned duplex turn-taking remains active. The
bundled RNN-T weights are not loaded.

Other checkpoints may carry an ASR head, a function head, both, or neither.
The heads are independent. When an ASR head is present, user transcription
and ASR-based forced turn-taking are available.

Leave both engine keys at ``native`` for PyTorch inference. Either component
can be switched to ``vllm_omni`` after converting a wrapper checkpoint.

Programmatic Usage
^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from nemo.collections.speechlm2.inference import S2SPipelineBuilder
    from nemo.collections.speechlm2.inference.model_wrappers.engine_selection import (
        inference_precision_from_cfg,
    )

    with inference_precision_from_cfg(cfg.s2s):
        pipeline = S2SPipelineBuilder.build_pipeline(cfg)
        try:
            outputs = pipeline.run(audio_filepaths, options=options)
        finally:
            pipeline.shutdown()

    # returns list[S2SStreamingOutput], one per input file
    # each element has: .output_text_str, .output_asr_text_str, .audio_filepath, ...

File Inputs and Manifests
^^^^^^^^^^^^^^^^^^^^^^^^^

The ``audio_file`` argument accepted by
``examples/speechlm2/nemo_inference_pipelines/s2s_streaming_infer.py`` may be:

* A single ``.wav`` file.
* A directory, in which case all ``.wav`` files in that directory are streamed.
* A line-delimited ``.json`` or ``.jsonl`` manifest listing audio files.

Manifest entries must provide ``audio_filepath`` and may also provide
``system_prompt`` and ``text``:

.. code-block:: json

    {"audio_filepath": "audio/example.wav", "system_prompt": "You are helpful.", "text": "reference user transcript"}

The JSON/JSONL manifest accepted by ``s2s_streaming_infer.py`` has this schema.
Its ``text`` field is read only as an optional reference transcript for WER on
the ASR/user side.  Generated agent text is produced by the model and written to
``pred_text`` in the output JSON.

This lightweight streaming inference manifest is distinct from the dataset
manifests used for SpeechLM2 training and offline evaluation.  For those dataset
formats, see :doc:`SpeechLM2 datasets <datasets>`.

File paths in streaming inference manifests are resolved relative to the
manifest file.  Audio from file inputs is converted to mono and
resampled to ``streaming.input_sample_rate`` before it is chunked.


Configuration
-------------

The streaming inference configuration is defined in
``examples/speechlm2/nemo_inference_pipelines/conf/s2s_streaming.yaml``.

Key configuration groups:

S2S Model Settings (``s2s``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Parameter
     - Default
     - Description
   * - ``model_path``
     - (required)
     - Path to the NemotronVoiceChat HuggingFace checkpoint.
   * - ``llm_engine_type``
     - ``native``
     - LLM backend: ``native`` or ``vllm_omni``.
   * - ``tts_engine_type``
     - ``native``
     - TTS backend: ``native`` or ``vllm_omni``. Independent of the LLM.
   * - ``speaker_name``
     - ``null``
     - Required when ``decode_audio`` is true. Must match a speaker registered
       in the checkpoint. Public checkpoints do not support cloning from a
       reference wav.
   * - ``system_prompt``
     - (required)
     - Text injected into the LLM KV cache before audio streaming begins.
   * - ``compute_dtype``
     - ``bfloat16``
     - Precision for LLM/embedding layers.
   * - ``use_perception_cache``
     - ``true``
     - Cache-aware streaming for the perception encoder.
   * - ``use_llm_cache``
     - ``false``
     - Reuse the native LLM KV cache instead of replaying history. NemotronH
       requires Transformers 5.13 or newer; leave disabled on older runtimes.
   * - ``top_p``
     - ``0.5``
     - Top-p sampling threshold.
   * - ``temperature``
     - ``0.3``
     - Sampling temperature.
   * - ``repetition_penalty``
     - ``1.1``
     - Repetition penalty applied to previously generated tokens.
   * - ``deterministic``
     - ``false``
     - Force deterministic mode (native engine only).
   * - ``profile_timing``
     - ``false``
     - Insert ``torch.cuda.synchronize()`` around each stage for accurate
       per-stage timing.  Disabled by default to avoid GPU stalls.

Streaming Settings (``streaming``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Parameter
     - Default
     - Description
   * - ``chunk_size_in_secs``
     - (required)
     - Audio processed per inference step.  Must be a multiple of 0.08 s.
   * - ``buffer_size_in_secs``
     - (required)
     - Sliding-window size passed to the perception encoder.
   * - ``batch_size``
     - ``1``
     - Must be ``1``. Single-stream inference only.
   * - ``max_len``
     - ``8192``
     - Maximum number of frames per stream.

Padding Settings (top-level)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These append trailing silence after the real input so the agent is more likely
to finish speaking before the stream ends. They are not batch padding. At most
one may be set:

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Parameter
     - Default
     - Description
   * - ``pad_audio_to_sec``
     - ``null``
     - Append trailing silence so each input reaches this duration.
   * - ``pad_silence_ratio``
     - ``null``
     - Append trailing silence equal to this fraction of the original duration.
   * - ``pad_audio_by_sec``
     - ``null``
     - Append this many extra seconds of trailing silence.


Server Integration
------------------

Use ``generate_step()`` directly when audio does not come from complete files:
for example, from a microphone, socket, Triton server, or browser UI.  The
caller owns the input connector: capture audio, convert it to mono
``streaming.input_sample_rate`` samples, split it into chunks, and pass those
chunks as ``Frame`` objects.

``generate_step()`` returns one ``GenerateStepOutput`` for each input frame,
containing the audio, agent text, and user ASR text produced by that step.

.. code-block:: python

    from nemo.collections.asr.inference.streaming.framing.request import Frame
    from nemo.collections.speechlm2.inference import S2SRequestOptions

    # 1. Initialize the stream before recording starts.
    #    Send empty audio because prefill will likely take longer than chunk_size_in_secs.
    init_frame = Frame(
        samples=torch.empty(0),
        stream_id=stream_id,
        is_first=True, is_last=False,
        options=S2SRequestOptions(system_prompt=prompt, top_p=0.9),
    )
    pipeline.generate_step([init_frame])
    # -> the input connector can now start sending audio

    # 2. For each input audio chunk, run one streaming step.
    for chunk, is_last in audio_source:
        frame = Frame(
            samples=chunk,
            stream_id=stream_id,
            is_first=False, is_last=is_last,
        )
        outputs = pipeline.generate_step([frame])
        for out in outputs:
            send_to_client(out.audio, out.text, out.asr_text)

Per-stream options (``system_prompt``, ``top_p``, ``temperature``,
``repetition_penalty``) are attached to the ``is_first`` frame via
``S2SRequestOptions``.  Any field left as ``None`` falls back to the
pipeline-level YAML default through ``fill_defaults()``.

.. _init-and-latency:

Init and Latency
^^^^^^^^^^^^^^^^

When ``generate_step`` sees ``is_first``, it always runs stream
initialization (context creation, KV-cache prefill).  If the frame also
carries audio, inference runs immediately after init in the same call.

For **latency-sensitive** integrations, prefill can take hundreds of
milliseconds or even multiple seconds.  Send ``is_first`` with **empty audio**,
wait for the response confirming init is done, and only then start sending real
audio.  This prevents input audio from queuing up during the expensive prefill
phase.

For **batch/offline** usage (``run()``), there is no real-time
constraint.  The first frame carries both ``is_first`` and real audio,
so init and first-chunk processing happen in one call with no extra
round-trip.

The pipeline makes no distinction between these cases — it initializes
on ``is_first`` and processes whatever audio is present.  The latency
trade-off is entirely the caller's choice.


Architecture
------------

File Chunking in ``run()``
^^^^^^^^^^^^^^^^^^^^^^^^^^

For file inputs, ``StreamingS2SPipeline.run()`` uses
``SilencePaddedContinuousBatchedFrameStreamer`` to load the audio paths,
convert them to mono ``streaming.input_sample_rate`` audio, and emit ``Frame``
chunks.  The streamer uses the configured chunk size, batch size, and optional
silence-padding settings.  ``run()`` then passes each emitted frame batch to
``generate_step``.  Live integrations do not need this helper; they can
construct ``Frame`` objects directly and call ``generate_step``.

The Core Streaming Loop
^^^^^^^^^^^^^^^^^^^^^^^

``StreamingS2SPipeline.run()`` orchestrates the streaming loop, delegating
per-chunk inference to ``generate_step()`` and saving outputs as streams
finish.  In simplified pseudocode:

.. code-block:: python

    # Inside StreamingS2SPipeline.run() (simplified):
    self.open_session()
    for frames in streamer:
        # step_outputs[i] carries GenerateStepOutput.audio / .text / .asr_text
        # — the new agent audio and text produced by this chunk.
        step_outputs = self.generate_step(frames)
        self._finalize_and_save_finished_streams(frames, ...)
    self.close_session()
    # run() then returns list[S2SStreamingOutput], one per input file

``run()`` returns a list of finalized ``S2SStreamingOutput`` objects (one per
input audio file) with the accumulated texts, token tensors, and audio
filepaths.

``run()`` writes outputs as each stream finishes, so results appear on disk
before the full run completes.  For each stream:

* ``<stem>.txt`` - agent transcript.
* ``<stem>.ctm`` & ``<stem>_asr.ctm`` - per-token timing for agent text and ASR text.
  Timestamps reflect when the text token was generated by the model.
* ``<stem>.wav`` & ``<stem>_input_output.wav`` - generated agent audio, plus a
  stereo file with input on one channel and output on the other.

  * In the stereo file, the generated-output channel is offset by one chunk so
    playback reflects the minimum delay from waiting for a full input chunk
    before generating output (Note: actual inference time would add to this in
    a real deployment).
  * Both audio files are skipped when ``s2s.decode_audio=false``.

After all streams finish, ``s2s_streaming_infer.py`` also writes two JSON
summaries of the run: ``output_raw.json`` (full token stream including padding
tokens) and ``output_processed.json`` (padding tokens removed for legibility).

``run()`` loops over chunks of existing audio files, calling ``generate_step()``
on each; ``generate_step()`` can also be called directly when audio comes from
a non-file source.


What Happens Inside One Step
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

    generate_step(frames)
        │
        ├─ for each frame where is_first=True:
        │      │
        │      └─ _init_state(stream_id, options)
        │             1. fill_defaults()           ← fill None fields from YAML
        │             2. create_state(options)      ← pipeline-level state
        │             3. reset context_manager       ← fresh decode-state storage
        │             4. prefill system prompt      ← populate LLM KV cache
        │
        └─ any frames with audio?
               │
              NO → return empty outputs  (server prefill-only request)
               │
             YES → update per-stream sliding audio buffer
                      │
                      ▼
                   generate_step_for_frames()
                      1. perception encoder
                      2. per-frame LLM loop
                      3. per-frame TTS (when decode_audio=true)
                      4. codec decode  (when decode_audio=true)
                      5. state updates + output accumulation
                      6. return list[GenerateStepOutput]

Each call to ``generate_step(frames)`` performs:

1. **Stream init on** ``is_first`` -- If a frame has ``is_first=True``, the
   private ``_init_state()`` method runs: per-stream options are merged with
   pipeline defaults (via ``S2SRequestOptions.fill_defaults()``),
   a fresh ``S2SStreamingOutput`` is created, the context manager is
   allocated, and the LLM KV cache is prefilled with the system prompt and
   TTS speaker embedding.  This mirrors ASR's ``init_state()`` called inside
   ``transcribe_step()``.  If the frame carries no audio (zero-length
   samples), the method returns after init — this is the recommended
   pattern for latency-sensitive deployments (see
   :ref:`init-and-latency` above).

2. **Audio buffer update** -- ``generate_step`` updates each stream's rolling
   audio buffer so the model receives the current ``buffer_size_in_secs``-size
   window of audio.

3. **Model inference** via ``infer_one_step(audio_buffer, state)``:

   a. **Perception** -- The audio buffer is encoded by the streaming
      FastConformer encoder into frame embeddings.
   b. **Per-frame LLM loop** -- For each of the ``num_frames_per_chunk``
      frames, the pipeline builds an input embedding (user audio +
      previous-step text/ASR tokens), runs it through the LLM, and obtains
      predicted text and ASR tokens.
   c. **TTS code generation** -- When ``s2s.decode_audio=true``, predicted text
      tokens are fed into the EarTTS model to produce audio codec codes.
   d. **Codec decode** -- When ``s2s.decode_audio=true``, the accumulated codes
      are decoded into a waveform.

4. **State updates** -- The per-stream ``StreamingDecodeState`` is updated
   with model-side decode state such as generated-token history and caches.

5. **Output accumulation** -- Decoded audio and text are appended to the
   per-stream ``S2SStreamingOutput``.


Data Objects
^^^^^^^^^^^^

The streaming pipeline uses four data objects.  Two are **model-level**
(owned by the model wrapper) and two are **pipeline-level** (owned by
``StreamingS2SPipeline``):

.. code-block:: text

    Model level (decode_state.py)
    ─────────────────────────────────────────────────────────────
    StreamingDecodeState          created per stream
      GPU KV caches, token          mutated in-place by infer_one_step()
      workspaces, perception        destroyed at end-of-stream
      cache, codec cache
              │
              │ infer_one_step()
              ▼
    InferenceStepResult           created each step
      predicted tokens, text        returned to the pipeline
      strings, decoded audio        consumed immediately

    Pipeline level (streaming_s2s_pipeline.py, s2s_streaming_output.py)
    ─────────────────────────────────────────────────────────────
    S2SStreamingOutput            created per stream
      accumulates audio chunks      finalized fields (text_with_timestamps,
      and text across steps         audio_filepath, etc.) filled at end-of-stream
                                    returned by run()
              ▲
              │ each step appends
              │
    GenerateStepOutput            created each step
      incremental per-stream        returned by generate_step()
      audio + text                  used by server integrations

**StreamingDecodeState** lives in ``S2SContextManager`` and holds the heavy
GPU tensors (KV caches, perception cache, token workspaces).  It is created
by the model wrapper, mutated in-place by ``infer_one_step()``, and
destroyed at end-of-stream.

**S2SStreamingOutput** lives in the pipeline's ``_state_pool``.  During
streaming it accumulates audio chunks and text parts.  At end-of-stream the
pipeline populates its finalized fields (``text_with_timestamps``,
``raw_text``, ``audio_filepath``, token tensors) and returns the same
object from ``run()``.


Inference Backends
^^^^^^^^^^^^^^^^^^

NemotronVoiceChat has two inference components that each need a backend:

- **LLM** (DuplexSTT backbone) -- takes audio embeddings from the perception
  encoder and predicts text tokens plus checkpoint-dependent ASR and function
  tokens at each frame.
- **TTS** (EarTTS) -- takes the predicted text token and produces audio codec
  codes (RVQ acoustic tokens).

Two engines can drive those components. ``llm_engine_type`` and
``tts_engine_type`` select them independently; an omitted key defaults to
``native``:

.. list-table::
   :header-rows: 1
   :widths: 25 25 25

   * - ``llm_engine_type``
     - ``tts_engine_type``
     - Components
   * - ``native``
     - ``native``
     - ``PyTorchLLM`` + ``PyTorchEarTTS``
   * - ``native``
     - ``vllm_omni``
     - ``PyTorchLLM`` + one-stage EarTTS ``AsyncOmni``
   * - ``vllm_omni``
     - ``native``
     - one-stage Nemotron ``AsyncOmni`` + ``PyTorchEarTTS``
   * - ``vllm_omni``
     - ``vllm_omni``
     - independent one-stage Nemotron and EarTTS ``AsyncOmni`` engines

The two components have one contract each, and each contract has a PyTorch and
a vLLM-Omni implementation:

.. code-block:: text

    backend/
        llm.py                  # DuplexLLM  ABC: step() -> this frame's tokens
        eartts.py               # DuplexTTS  ABC: step() -> this frame's codes
        pytorch/
            llm.py              # PyTorchLLM     (wraps the DuplexSTT forward pass)
            eartts.py           # PyTorchEarTTS  (wraps DuplexEARTTS.infer_codes_one_step)
        vllm/
            llm.py              # VllmLLM        (NotImplementedError in this PR)
            eartts.py           # VllmEarTTS     (NotImplementedError in this PR)

This PR ships the native engines. ``VllmLLM`` and ``VllmEarTTS`` exist so the
wrapper's frame loop already matches the combined form; selecting
``llm_engine_type`` / ``tts_engine_type`` of ``vllm_omni`` raises
``NotImplementedError`` at construction. The runtime (``inference/vllm_omni/``,
``OmniRuntime``, wrapper-checkpoint conversion) is the parent commit on
``duplex-vllm-omni-on-main``.

``NemotronVoicechatInferenceWrapper`` selects one implementation per component
at construction and stores them as ``llm_backend`` and ``tts_backend``. Its
frame loop then calls ``step()`` on each without inspecting the engine type, so
all four combinations run the same code path. Perception, the audio codec and
tokenization stay on PyTorch in every combination.

The contracts hold only ``step()``. Cache creation, prompt prefill and request
abort are PyTorch-only -- vLLM does them inside the engine and the
session -- so they stay on the PyTorch classes, which the wrapper reaches
through ``model_llm_interface`` / ``model_eartts_interface`` during stream
setup. Those two attributes are ``None`` when their component runs on vLLM.

Text sampling (top-p, repetition penalty, temperature) is shared rather than
duplicated: both backends decode the text head with
``inference.model_wrappers.text_sampling.sample_text_token``. ``PyTorchLLM``
calls it directly; the vLLM path reaches it through
``SharedTextSamplingLogitsProcessor``.

Config support by backend
"""""""""""""""""""""""""

Settings that model code reads off its own config are listed in
``inference.model_wrappers.config_overrides``, which records where each one
lands and which backends honour it. Anything a selected backend ignores is
reported at load time rather than silently doing nothing.

.. list-table::
   :header-rows: 1
   :widths: 32 12 12 44

   * - Setting
     - native
     - vllm_omni
     - Notes
   * - ``inference_pad/bos/eos_boost``
     - yes
     - yes
     - Agent text channel. vLLM applies them in the shared sampling hook.
   * - ``inference_user_pad/bos/eos_boost``
     - yes
     - yes
     - ASR channel. Its logits never reach vLLM's sampler, so the converted
       Nemotron applies them itself; the wrapper writes the values into
       ``nemotron/config.json`` before the engine starts.
   * - ``force_turn_taking`` (+ threshold, pad window)
     - yes
     - yes
     - The rewritten text token is fed back explicitly, so Nemotron's history
       stays consistent with the text channel.
   * - ``inference_force_speech_silence_on_eos``
     - yes
     - partial
     - Applied inside EarTTS on both paths. The converted EarTTS always
       substitutes codec silence on EOS and has no flag, so it cannot honour
       ``false``.
   * - ``inference_top_p_or_k``, ``inference_noise_scale``, ``inference_guidance_scale``
     - yes
     - no
     - The vLLM EarTTS takes sampling from the converted checkpoint and
       ``vllm_omni_config`` instead.
   * - ``deterministic``
     - yes
     - no
     - Rejected: vLLM's kernels have no deterministic mode.
   * - ``use_llm_cache``, ``use_tts_torch_compile``, ``use_tts_subword_cache``
     - yes
     - n/a
     - Performance knobs whose intent vLLM already meets: it always keeps a
       paged KV cache, compiles inside vLLM, and bakes the subword table at
       conversion. Setting them warns.

vLLM-Omni Integration
"""""""""""""""""""""

Selecting ``vllm_omni`` is rejected at wrapper construction in this PR. The
intended shape is: each component selected as ``vllm_omni`` gets the ``Vllm*``
implementation of its contract, and the corresponding native class is not
created. The wrapper starts only the selected one-stage ``AsyncOmni`` engine
or engines. That runtime lives on the parent commit
(``duplex-vllm-omni-on-main``):

- **Nemotron** -- ``NemotronDuplexHForCausalLM``, which consumes the per-step
  acoustic embedding and samples a text token plus the checkpoint's optional
  ASR or function token.
- **EarTTS** -- ``EarTTSForCausalLM``, which receives each sampled text token
  from NeMo and emits one acoustic frame.

The split keeps the component boundary in NeMo, so either component can be
replaced without changing the other engine. Nemotron settings come from
``inference/vllm_omni/deploy/nemotron_voicechat.yaml`` and EarTTS settings from
``inference/vllm_omni/deploy/eartts.yaml`` on that parent commit. Override them
independently with ``vllm_omni_config.stage_overrides`` and
``vllm_omni_config.eartts_stage_overrides``.

Nemotron text sampling uses vLLM's custom logits-processor hook to call the
same PyTorch sampling helper as the native backend. This preserves all-ones
greedy decoding, special-token bypass, top-p, temperature, and repetition
penalty over agent-frame history while leaving vLLM's built-in sampler in
greedy/no-penalty mode. Stochastic runs share the algorithm but are not
guaranteed to produce identical tokens across backends because their worker
processes do not share RNG state.

EarTTS classifier-free guidance uses two explicit requests in the same engine.
They have independent KV caches, but a custom scheduler advances them in
lockstep. The unconditional request replaces text conditioning with the
checkpoint's ``null_emb``; the MaskGIT sampler applies the native guidance
formula and returns only the conditional stream's codes. Configure it with
``vllm_omni_config.guidance_enabled`` and ``guidance_scale``.

Perception, the audio codec and tokenization stay on PyTorch in every engine pairing.

Auxiliary channels
''''''''''''''''''

Besides the agent text token, a checkpoint may predict ASR and/or function
tokens per frame. The flags are independent:

- ``predict_user_text=True`` gives an ASR channel (``asr_head`` plus its own
  ``embed_asr_tokens`` table) that transcribes the user.
- ``use_function_head=True`` gives a function channel (``function_head``),
  whose feedback is embedded with the *text* ``embed_tokens`` and scaled by
  ``duplex_function_channel_weight``.

The previous frame's auxiliary token is part of the next frame's model input,
so it remains in autoregressive state to reproduce the checkpoint's text and
audio. The pipeline also exposes a decoded copy to clients. Function-channel
text is informational; the pipeline does not execute tool calls.

Both engines support both channels. On ``vllm_omni`` the converter
(``convert_duplex_stt_checkpoint.py``) records which heads a checkpoint carries
as ``use_asr_head`` / ``use_function_head`` in the wrapper config, because
Nemotron has to decide which modules to build before it sees any weights, and
then feeds each enabled channel back to itself through its
``postprocess`` -> ``preprocess`` buffers.

On stock vLLM-Omni 0.26, the registered Nemotron stage keeps
``final_output_type="text"`` while using the final multimodal engine-output
path for its optional auxiliary tensor. Text remains on ``RequestOutput`` and
``asr_tokens`` or ``function_tokens`` reaches the caller beside it. This
configuration was validated with the public function-head VoiceChat
checkpoint; ASR-head checkpoints use the same converter and extraction path.

Client APIs keep the channels distinct: ASR-head checkpoints populate the
existing ASR output, while function-head checkpoints populate the function
output and capability metadata. A missing head always produces an empty
corresponding client field.
