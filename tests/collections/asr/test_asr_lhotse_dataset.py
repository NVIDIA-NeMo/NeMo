import pytest
import torch
from lhotse import CutSet, SupervisionSegment
from lhotse.testing.dummies import DummyManifest

from nemo.collections.asr.data.audio_to_text_lhotse import LhotseSpeechToTextBpeDataset
from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer, create_spt_model


@pytest.fixture(scope="session")
def tokenizer(tmp_path_factory) -> SentencePieceTokenizer:
    tmpdir = tmp_path_factory.mktemp("klingon_tokens")
    text_path = tmpdir / "text.txt"
    text_path.write_text("\n".join(map(chr, range(ord('a'), ord('z')))))
    model_path, vocab_path = create_spt_model(
        text_path, vocab_size=32, sample_size=-1, do_lower_case=False, output_dir=str(tmpdir)
    )
    return SentencePieceTokenizer(model_path)


def test_lhotse_asr_dataset(tokenizer):
    # 3 cuts of duration 1s with audio and a single supervision with text 'irrelevant'
    cuts = DummyManifest(CutSet, begin_id=0, end_id=3, with_data=True)

    # cuts[0] is the default case: audio + single untokenized superivision

    # cuts[1]: audio + single pre-tokenized superivision
    cuts[1].supervisions[0].tokens = tokenizer.text_to_ids(cuts[1].supervisions[0].text)

    # cuts[2]: audio + two supervisions
    cuts[2].supervisions = [
        SupervisionSegment(id="cuts2-sup0", recording_id=cuts[2].recording_id, start=0, duration=0.5, text="first"),
        SupervisionSegment(id="cuts2-sup1", recording_id=cuts[2].recording_id, start=0.5, duration=0.5, text="second"),
    ]

    dataset = LhotseSpeechToTextBpeDataset(tokenizer=tokenizer)
    batch = dataset[cuts]

    assert isinstance(batch, tuple)
    assert len(batch) == 4
    assert all(isinstance(t, torch.Tensor) for t in batch)

    audio, audio_lens, tokens, token_lens = batch

    assert audio.shape == (3, 16000)
    assert audio_lens.tolist() == [16000] * 3

    assert tokens.shape == (3, 13)
    assert tokens[0].tolist() == [1, 10, 19, 19, 6, 13, 6, 23, 2, 15, 21, 0, 0]
    assert tokens[1].tolist() == tokens[0].tolist()
    assert tokens[2].tolist() == [1, 7, 10, 19, 20, 21, 1, 20, 6, 4, 16, 15, 5]

    assert token_lens.tolist() == [11, 11, 13]
