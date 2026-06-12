NeMo master RNNT inference script: examples/asr/asr_chunked_inference/rnnt/speech_to_text_streaming_infer_rnnt.py
Dependent label-looping decoder folder: nemo/collections/asr/parts/submodules/transducer_decoding/ (rnnt_label_looping.py, label_looping_base.py, batched_hyps.py)

Label-looping decoder (rnnt_label_looping.py): outer `while active_mask.any()` steps through encoder frames; inner `while advance_mask.any()` emits all non-blank tokens from one frame before advancing time_indices to the next.
Our implementation mirrors this inside `_rnnt_step()`: `is_blank` is set from the first joint prediction per frame (unchanged turn-taking signal for blank_count/nonblank_consec); the inner loop then calls joint+predictor repeatedly until blank, appending each token to `_emitted` and advancing `_cur_pred_out`/`_cur_pred_hidden`.
Emitted tokens are appended to `y_sequence` in the RNNT state dict each frame; `_rnnt_decode_text(y_sequence)` decodes them via `rnnt_joint.vocabulary` (SentencePiece, ▁=word boundary) and writes to `state.output_asr_text_str` in the pipeline.
Both FC (live_asr_emb_frame path, _run_fc_async_steps line ~1537) and non-FC (asr_emb path, infer_one_step line ~2246) call the same `_rnnt_step()` — y_sequence accumulates across both modes in one list per turn.
Model: cand7 fine-tuned ASR checkpoint at S2S_PRETRAINED_ASR — same weights as turn-taking, no extra VRAM, _rnnt_max_symbols=10 (configurable via model_cfg["rnnt_max_symbols"]).
