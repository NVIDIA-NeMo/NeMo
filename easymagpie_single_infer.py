"""Minimal single-utterance inference script for EasyMagpieTTS."""
import argparse
import os

os.environ['OMP_NUM_THREADS'] = '2'

import soundfile as sf
from omegaconf import open_dict
from nemo.collections.tts.models import EasyMagpieTTSModel


def main():
    p = argparse.ArgumentParser(description="EasyMagpieTTS single inference")
    p.add_argument("--transcript", required=True, help="Text to synthesize")
    p.add_argument("--model_path", required=True, help="Path to .nemo TTS checkpoint")
    p.add_argument("--codec_model_path", required=True, help="Path to .nemo codec model")
    p.add_argument("--output_path", required=True, help="Output .wav file path")

    ctx = p.add_mutually_exclusive_group(required=True)
    ctx.add_argument("--context_audio_path", help="Path to context audio file")
    ctx.add_argument("--context_text", help="Context text string")

    p.add_argument("--phoneme_tokenizer_path", default=None,
                   help="Path to phoneme tokenizer JSON (auto-detected next to this script if omitted)")
    p.add_argument("--language", default="en", choices=["en", "zh", "es", "fr", "de", "it", "hi", "vi"])
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--topk", type=int, default=80)
    p.add_argument("--use_cfg", action="store_true", default=True)
    p.add_argument("--no_cfg", dest="use_cfg", action="store_false")
    p.add_argument("--cfg_scale", type=float, default=2.5)
    p.add_argument("--max_steps", type=int, default=300)
    args = p.parse_args()

    phoneme_tokenizer_path = args.phoneme_tokenizer_path
    if phoneme_tokenizer_path is None:
        phoneme_tokenizer_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "bpe_ipa_tokenizer_2048_en_de_es_fr_hi_it_vi_zh.json",
        )

    # --- load model ---
    model_cfg = EasyMagpieTTSModel.restore_from(args.model_path, return_config=True)
    with open_dict(model_cfg):
        model_cfg.codecmodel_path = args.codec_model_path
        model_cfg.train_ds = None
        model_cfg.validation_ds = None
        if getattr(model_cfg, "phoneme_tokenizer", None) is not None:
            model_cfg.phoneme_tokenizer.tokenizer_path = phoneme_tokenizer_path

    model = EasyMagpieTTSModel.restore_from(args.model_path, override_config_path=model_cfg)
    model.use_kv_cache_for_inference = True
    model.eval().cuda().float()

    # --- resolve context ---
    if args.context_audio_path:
        use_language_tag = bool(getattr(model, "add_language_to_context_text", False))
        context_text = f"[{args.language.upper()}]" if use_language_tag else "[NO TEXT CONTEXT]"
        context_audio_path = args.context_audio_path
    else:
        context_text = args.context_text
        context_audio_path = None

    transcript = args.transcript.strip()
    if not transcript.endswith((".", "?", "!")):
        transcript += "."

    # --- infer ---
    audio, audio_len = model.do_tts(
        transcript=transcript,
        context_audio_file_path=context_audio_path,
        context_text=context_text,
        use_cfg=args.use_cfg,
        cfg_scale=args.cfg_scale,
        use_local_transformer=True,
        temperature=args.temperature,
        topk=args.topk,
        max_steps=args.max_steps,
    )

    audio_np = audio[0, : audio_len[0]].cpu().numpy()
    sf.write(args.output_path, audio_np, model.output_sample_rate)
    print(f"Saved {args.output_path}  ({audio_np.shape[0]} samples, {model.output_sample_rate} Hz)")


if __name__ == "__main__":
    main()
