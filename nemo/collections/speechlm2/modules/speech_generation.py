# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
from omegaconf import DictConfig
from torch import Tensor, nn
import torchaudio

from nemo.collections.tts.modules import transformer_2501
from nemo.core.classes.module import NeuralModule


def build_vocabs(subword_vocab_items):
    # tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer_name)
    # subword_vocab_items = tokenizer.vocab.items()
    org_char_vocab = {subword: subword_id for subword, subword_id in subword_vocab_items if len(subword) == 1}
    sorted_char_vocab = dict(sorted(org_char_vocab.items(), key=lambda x: x[1]))
    char_vocab = {k: i for i, (k, _) in enumerate(sorted_char_vocab.items())}
    assert sorted(char_vocab.values()) == list(range(len(char_vocab)))
    subword_id_to_char_ids = {
        subword_id: tuple(char_vocab[char] for char in subword) for subword, subword_id in subword_vocab_items
    }

    assert max(subword_id_to_char_ids) == len(subword_id_to_char_ids) - 1
    # add padding_idx
    subword_padding_idx = len(subword_id_to_char_ids)
    subword_id_to_char_ids[subword_padding_idx] = (len(char_vocab),)

    return subword_id_to_char_ids, char_vocab, subword_padding_idx


def sequence_mask(lengths: Tensor, max_length: int | None = None) -> Tensor:
    if max_length is None:
        max_length = int(lengths.max().item())
    x = torch.arange(max_length, dtype=lengths.dtype, device=lengths.device)
    return x.unsqueeze(0) < lengths.unsqueeze(1)


class CharAwareSubwordEncoder(NeuralModule):
    def __init__(self, params: DictConfig, llm_tokenizer_vocab_items: dict = None):
        super().__init__()
        self.subword_id_to_char_ids, self.char_vocab, self.subword_padding_idx = build_vocabs(llm_tokenizer_vocab_items)
        self.embed_tokens = nn.Embedding(self.vocab_size + 1, params["d_model"], padding_idx=self.vocab_size)
        self.encoder = transformer_2501.Transformer(**params)

    @property
    def vocab_size(self):
        return len(self.char_vocab)

    def prepare_inputs(self, subword_ids: Tensor, padding_mask: Tensor) -> tuple[Tensor, Tensor]:
        device = subword_ids.device

        subword_id_list = torch.masked_select(subword_ids, padding_mask).cpu().tolist()
        char_id_list = [list(self.subword_id_to_char_ids[x]) for x in subword_id_list]

        char_lengths = torch.tensor([len(x) for x in char_id_list], dtype=torch.long, device=device)
        batch_size = char_lengths.size(0)

        char_ids = torch.full((batch_size, int(char_lengths.max().item())), self.vocab_size, dtype=torch.long)
        for i in range(batch_size):
            char_ids[i, : char_lengths[i]] = torch.tensor(char_id_list[i])
        char_ids = char_ids.to(device=device)
        return char_ids, char_lengths

    def forward(self, subword_ids: Tensor, subword_mask: Tensor | None = None) -> Tensor:
        device = subword_ids.device
        if subword_mask is None:
            subword_mask = torch.ones_like(subword_ids).bool()
        else:
            subword_mask = subword_mask.bool()

        if subword_mask.ndim == 3:
            subword_mask = subword_mask.squeeze(-1)

        char_ids, char_lengths = self.prepare_inputs(subword_ids, subword_mask)
        char_mask = sequence_mask(char_lengths)
        char_emb = self.embed_tokens(char_ids)
        # char emb has the shape  [B*T, N, channels], where N is the max number of chars tokens decoded from bpe tokens
        x = self.encoder(
            x=char_emb,
            x_mask=char_mask
        )['output']

        # Get average embedding over the chars
        mean_emb = ((x / char_mask.unsqueeze(-1).sum(1, keepdim=True)) * char_mask.unsqueeze(-1)).sum(1)
        subword_emb = torch.zeros((subword_mask.size(0), subword_mask.size(1), mean_emb.size(-1)), device=device)
        subword_emb[subword_mask.unsqueeze(-1).expand(-1, -1, mean_emb.size(-1))] = mean_emb.view(-1)

        return subword_emb


class TransformerARSpeechDecoder(NeuralModule):
    def __init__(
        self,
        speech_decoder_parms: DictConfig,
        lantent_dim: int,
        num_audio_codebooks: int,
        num_audio_tokens_per_codebook: int,
    ):
        super().__init__()
        self.use_input_cache = False
        self.speech_decoder_parms = speech_decoder_parms
        self.lantent_dim = lantent_dim
        self.num_audio_codebooks = num_audio_codebooks
        self.num_audio_tokens_per_codebook = num_audio_tokens_per_codebook
        # optional configs
        self.cfg_unconditional_prob = self.speech_decoder_parms.pop("cfg_unconditional_prob", None)
        self.cfg_scale = self.speech_decoder_parms.pop("cfg_scale", 2.5)
        self.cond_on_prev_audio_tokens = self.speech_decoder_parms.pop("cond_on_prev_audio_tokens", False)
        self.detach_input = self.speech_decoder_parms.pop("detach_input", False)
        self.cond_on_text_tokens = self.speech_decoder_parms.pop("cond_on_text_tokens", False)
        self.cond_on_llm_latent = self.speech_decoder_parms.pop("cond_on_llm_latent", True)
        self.cond_on_speech_encoder_emb = self.speech_decoder_parms.pop("cond_on_speech_encoder_emb", False)
        self.use_speaker_encoder = self.speech_decoder_parms.pop("use_speaker_encoder", False)
        self.speaker_embedding_dim = self.speech_decoder_parms.pop("speaker_embedding_dim", 192)
        self.inference_speaker_reference = self.speech_decoder_parms.pop("inference_speaker_reference", None)
        self.max_speaker_reference_len = self.speech_decoder_parms.pop("max_speaker_reference_len", 5)
        self.speaker_encoder_model_name = self.speech_decoder_parms.pop("speaker_encoder_model_name", 'titanet_large')
        self.cond_on_char_embedding = self.speech_decoder_parms.pop("cond_on_char_embedding", False)
        self.speech_encoder_emb_quantizer_levels = self.speech_decoder_parms.pop("speech_encoder_emb_quantizer_levels", None)

        if self.use_speaker_encoder:
            # NeMo Speaker encoder
            self.speaker_encoder = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name=self.speaker_encoder_model_name)

            # freeze the pretrained speaker encoder
            self.speaker_encoder.eval()
            self.speaker_encoder.freeze()

            # speaker encoder projection
            self.speaker_encoder_emb_projection = nn.Linear(self.speaker_embedding_dim, self.speech_decoder_parms["d_model"])

            # generate a random speaker embedding
            inference_speaker_embedding = torch.randn([1, 1, self.speaker_embedding_dim])
            self.register_buffer("inference_speaker_embedding", inference_speaker_embedding)
            # if inference_speaker_reference is provided, replace random embedding by the reference speaker embedding
            if self.inference_speaker_reference:
                self.update_inference_speaker_embedding(self.inference_speaker_reference)

        if not self.cond_on_llm_latent and not self.cond_on_text_tokens and not self.cond_on_char_embedding:
            raise ValueError(
                "At least one of 'cond_on_text_tokens' or 'cond_on_llm_latent' or 'cond_on_char_embedding' must be True for the Speech Decoder."
            )

        if self.cond_on_text_tokens and self.cond_on_char_embedding:
            raise ValueError(
                "'cond_on_text_tokens' and 'cond_on_char_embedding' are incompatible, you need to select one of these options !!"
            )

        # projection to adapt llm embeddings into the same shape of speech decoder expected input
        if self.cond_on_llm_latent and lantent_dim != self.speech_decoder_parms["d_model"]:
            self.input_proj = nn.Linear(lantent_dim, self.speech_decoder_parms["d_model"])
        else:
            self.input_proj = None

        # instanciate T5-TTS decoder to full compatibility and potentialy load pretrained model
        self.t5_decoder = transformer_2501.Transformer(**self.speech_decoder_parms)

        # projection to predict audio codes
        self.final_proj = nn.Linear(self.speech_decoder_parms["d_model"], num_audio_codebooks * num_audio_tokens_per_codebook)

        if self.cond_on_char_embedding:
            self.cas_params = dict(self.speech_decoder_parms)
            # uses only 1 layer for the char encoder
            self.cas_params["n_layers"] = 1
            self.cas_encoder = CharAwareSubwordEncoder(self.cas_params, llm_tokenizer_vocab_items)

        # create embeddings for encode input tokens
        if self.cond_on_prev_audio_tokens:
            audio_embeddings = []
            for _ in range(self.num_audio_codebooks):
                audio_embeddings.append(nn.Embedding(num_audio_tokens_per_codebook, self.speech_decoder_parms["d_model"]))

            self.audio_embeddings = nn.ModuleList(audio_embeddings)

        if self.cond_on_text_tokens:
            self.text_embeddings = nn.Embedding(llm_vocab_size, self.speech_decoder_parms["d_model"])

        # if cond on llm latent create the projection to sum the embeddings
        if self.cond_on_llm_latent and (self.cond_on_text_tokens or self.cond_on_char_embedding):
            self.text_input_projection = nn.Linear(self.speech_decoder_parms["d_model"], self.speech_decoder_parms["d_model"])

        if self.cond_on_speech_encoder_emb:
            if self.speech_encoder_emb_quantizer_levels:
                from nemo.collections.tts.modules.audio_codec_modules import FiniteScalarQuantizer
                bottleneck_dim = len(self.speech_encoder_emb_quantizer_levels)
                self.speech_encoder_emb_quantizer_projection = nn.Linear(lantent_dim, bottleneck_dim)
                self.see_vector_quantizer = FiniteScalarQuantizer(self.speech_encoder_emb_quantizer_levels)
                self.speech_encoder_emb_projection = nn.Linear(bottleneck_dim, self.speech_decoder_parms["d_model"])
            else:
                self.speech_encoder_emb_projection = nn.Linear(lantent_dim, self.speech_decoder_parms["d_model"])

    @property
    def device(self):
        return next(self.parameters()).device

    def update_inference_speaker_embedding(self, audio_path):
        audio, sr = torchaudio.load(audio_path)
        audio_len = torch.tensor([audio.size(1)]).long()
        speaker_emb = self.get_speaker_embedding(audio.to(self.device), audio_len.to(self.device), sr)
        self.inference_speaker_embedding = speaker_emb.to(self.inference_speaker_embedding.dtype)

    def get_speaker_embedding(self, audio, audio_len, sr):
        # limit max audio len to avoid memory waste
        audio = audio[:, : int(self.max_speaker_reference_len*sr)]
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            with torch.no_grad():
                model_sr = self.speaker_encoder._cfg.train_ds.get('sample_rate', 16000)
                audio_resampled = torchaudio.functional.resample(audio, sr, model_sr)
                audio_len_resampled = audio_len * (model_sr / sr )
                _, g = self.speaker_encoder(input_signal=audio_resampled, input_signal_length=audio_len_resampled.long())
                g = g.unsqueeze(1)
        return g.to(audio.dtype)

    def forward(self, hidden_states, speech_mask, input_audio_tokens=None, target_text_tokens=None, speech_encoder_emb=None, speaker_encoder_emb=None, temperature=0.7, topk=80, greedy=True):
        # Megatron LLM parallel training returns T, B, F so reshape it
        # T, B, F = hidden_states.size()
        if hidden_states is not None:
            hidden_states = hidden_states.transpose(0, 1).contiguous() # .reshape(B, T, F) # from [T, B, F] to [B, T, F]
    
            if self.detach_input:
                hidden_states = hidden_states.detach()

        # input cache needed due our transformer kv cache implementation expect the whole left context
        if self.use_input_cache:
            if self.cache["hidden_states"] is None:
                self.cache["hidden_states"] = hidden_states
            else:
                self.cache["hidden_states"] = torch.cat([self.cache["hidden_states"], hidden_states], dim=1)
                hidden_states = self.cache["hidden_states"]

            if self.cache["speech_mask"] is None:
                self.cache["speech_mask"] = speech_mask
            else:
                self.cache["speech_mask"] = torch.cat([self.cache["speech_mask"], speech_mask], dim=1)
                speech_mask = self.cache["speech_mask"]

            if self.cache["input_audio_tokens"] is None:
                self.cache["input_audio_tokens"] = input_audio_tokens
            else:
                self.cache["input_audio_tokens"] = torch.cat([self.cache["input_audio_tokens"], input_audio_tokens], dim=1)
                input_audio_tokens = self.cache["input_audio_tokens"]

            if self.cache["target_text_tokens"] is None:
                self.cache["target_text_tokens"] = target_text_tokens
            else:
                if target_text_tokens is not None:
                    self.cache["target_text_tokens"] = torch.cat([self.cache["target_text_tokens"], target_text_tokens], dim=1)
                    target_text_tokens = self.cache["target_text_tokens"]


        # map hidden states to the shape of the
        if hidden_states is not None and self.input_proj is not None:
            speech_decoder_input = self.input_proj(hidden_states)
        else:
            speech_decoder_input = hidden_states

        # workaround for inference, because during inference speech_mask will be None
        if speech_mask is None:
            speech_mask = torch.ones((speech_decoder_input.size(0), speech_decoder_input.size(1))).to(speech_decoder_input.device)

        # if cond on text tokens, sum text tokens with the llm latent
        if self.cond_on_text_tokens and target_text_tokens is not None:
            if self.use_llm_text_emb:
                text_tokens_embedded = self.text_emb_projection(target_text_tokens)
            else:
                text_tokens_embedded = self.text_embeddings(target_text_tokens)

            # if cond_on_llm_latent use a projection to sum the embeddings
            if self.cond_on_llm_latent:
                speech_decoder_input = self.text_input_projection(speech_decoder_input)
                speech_decoder_input = speech_decoder_input + text_tokens_embedded
            else:
                speech_decoder_input = text_tokens_embedded

        if self.cond_on_char_embedding:
            # if inference time cache char_embs to speedup inference
            if self.use_input_cache:
                char_emb_last = self.cas_encoder(target_text_tokens[:, -1:], subword_mask=speech_mask[:, -1:])
                if self.cache["char_embs"] is None:
                    self.cache["char_embs"] = char_emb_last
                else:
                    self.cache["char_embs"] = torch.cat([self.cache["char_embs"], char_emb_last], dim=1)

                char_embs = self.cache["char_embs"]
            else:
                char_embs = self.cas_encoder(target_text_tokens, subword_mask=speech_mask)

            # if cond_on_llm_latent use a projection to sum the embeddings
            if self.cond_on_llm_latent:
                speech_decoder_input = self.text_input_projection(speech_decoder_input)
                speech_decoder_input = speech_decoder_input + char_embs
            else:
                speech_decoder_input = char_embs

        if self.cond_on_speech_encoder_emb:
            if self.detach_input:
                speech_encoder_emb = speech_encoder_emb.detach()

            
            if self.use_input_cache:
                if self.speech_encoder_emb_quantizer_levels:
                    speech_encoder_emb = self.speech_encoder_emb_quantizer_projection(speech_encoder_emb[:, -1:])
                    speech_encoder_emb, _ = self.see_vector_quantizer(inputs=speech_encoder_emb.transpose(1, 2), input_len=None)
                    speech_encoder_emb = speech_encoder_emb.transpose(1, 2)

                if self.cache["speech_encoder_emb"] is None:
                    self.cache["speech_encoder_emb"] = speech_encoder_emb
                else:
                    if speech_encoder_emb is not None:
                        self.cache["speech_encoder_emb"] = torch.cat([self.cache["speech_encoder_emb"], speech_encoder_emb], dim=1)
                        speech_encoder_emb = self.cache["speech_encoder_emb"]
            else:
                if self.speech_encoder_emb_quantizer_levels:
                    speech_encoder_emb = self.speech_encoder_emb_quantizer_projection(speech_encoder_emb)
                    speech_encoder_emb, _ = self.see_vector_quantizer(inputs=speech_encoder_emb.transpose(1, 2), input_len=None)
                    speech_encoder_emb = speech_encoder_emb.transpose(1, 2)

            speech_encoder_emb = self.speech_encoder_emb_projection(speech_encoder_emb)
            speech_decoder_input = speech_decoder_input + speech_encoder_emb

        if self.use_speaker_encoder:
            # for inference uses the inference cached speaker embedding
            # ToDo: replace the repeat operation by adding over all inference time steps to speedup
            if self.use_input_cache and not self.training:
                speaker_encoder_emb = self.inference_speaker_embedding
                # repeat speaker encoder embedding to match the time and batch dimention
                speaker_encoder_emb = speaker_encoder_emb.repeat(speech_decoder_input.size(0), speech_decoder_input.size(1), 1)
            else:
                # repeat speaker encoder embedding to match the time dimention
                if speaker_encoder_emb.size(1) != speech_decoder_input.size(1):
                    speaker_encoder_emb = speaker_encoder_emb.repeat(1, speech_decoder_input.size(1), 1)

            speaker_encoder_emb = self.speaker_encoder_emb_projection(speaker_encoder_emb)
            speech_decoder_input = speech_decoder_input + speaker_encoder_emb

        if self.cfg_unconditional_prob:
            if self.training:
                # if training drop the "text" conditioning in a percentage of batch
                if torch.rand(1).item() < self.cfg_unconditional_prob:
                    # make the whole batch zeros to the unconditional model
                    # ToDo: move it to cache to need to just create a 1 frame tensor in inference
                    speech_decoder_input = torch.zeros_like(speech_decoder_input)
            else:
                # if inference or evaluation create a zero tensor for speech decoder input and concatenate it to compute unconditional logits
                speech_decoder_input_zeros = torch.zeros_like(speech_decoder_input)
                speech_decoder_input = torch.cat([speech_decoder_input, speech_decoder_input_zeros], dim=0)
                # duplicate mask to match the new shape
                speech_mask = torch.cat([speech_mask, speech_mask], dim=0)
                # if cond on prev tokens enabled, so duplicate the tokens to the new shape
                if self.cond_on_prev_audio_tokens:
                    input_audio_tokens = torch.cat([input_audio_tokens, input_audio_tokens], dim=0)

        # audio tokens should not be dropped by cfg, so we keep it here
        if self.cond_on_prev_audio_tokens:
            if self.detach_input:
                input_audio_tokens = input_audio_tokens.detach()

            audio_tokens_embedded = self.embed_audio_tokens(input_audio_tokens.transpose(1, 2).contiguous()) # (B, T', E)
            speech_decoder_input = speech_decoder_input + audio_tokens_embedded

        decoder_out = self.t5_decoder(x=speech_decoder_input, x_mask=speech_mask)['output']

        # if it is true we need to return just the last autoregressive step, it is valid because for 1 frame input we produce 1 frame ouput
        if self.use_input_cache:
            decoder_out = decoder_out[:, -1:, :]

        # get the logits of all codebooks
        all_code_logits = self.final_proj(decoder_out)

        # if using cfg and it is in inference or evaluation mix unconditional and coditional logits
        if self.cfg_unconditional_prob and not self.training:
            batch_size = all_code_logits.size(0) // 2
            cond_logits = all_code_logits[:batch_size]
            uncond_logits = all_code_logits[batch_size:]
            all_code_logits = (1 - self.cfg_scale) * uncond_logits + self.cfg_scale * cond_logits

        # sample for inference
        if self.use_input_cache and not self.training:
            sampled_audio_tokens = self.sample_codes_from_logits(all_code_logits, temperature=temperature, topk=topk, greedy=greedy)
        else:
            sampled_audio_tokens = None

        return all_code_logits, sampled_audio_tokens

    def sample_codes_from_logits(self, all_code_logits_t, temperature=0.7, topk=80, greedy=True):
        # all_code_logits_t: (B, num_codebooks * num_tokens_per_codebook), logits at a given timestep
        all_preds = []
        for idx in range(self.num_audio_codebooks):
            si = idx * self.num_audio_tokens_per_codebook
            ei = si + self.num_audio_tokens_per_codebook
            codebook_logits = all_code_logits_t[:, :, si:ei] # (B, num_tokens_per_codebook)
            B, T = codebook_logits.size(0), codebook_logits.size(1)
            if greedy:
                codebook_preds = torch.argmax(codebook_logits, dim=-1)
            else:
                codebook_logits_topk = torch.topk(codebook_logits, topk, dim=-1)[0] # (B, topk)
                codebook_probs = torch.softmax(codebook_logits / temperature, dim=-1) # (B, num_tokens_per_codebook)
                codebook_preds = torch.multinomial(codebook_probs, 1) # (B, 1)
            all_preds.append(codebook_preds.view(B, T).contiguous().long()) # T, B to be compatible with megatron

        return all_preds

    def embed_audio_tokens(self, audio_tokens):
        # Add and average the embeddings of the audio tokens across the codebooks
        audio_embedding = None
        for c in range(self.num_audio_codebooks):
            embedding = self.audio_embeddings[c](audio_tokens[:, c, :])
            if audio_embedding is None:
                audio_embedding = embedding
            else:
                audio_embedding = audio_embedding + embedding
        audio_embedding = audio_embedding / audio_tokens.size(1)
        return audio_embedding

    def reset_input_and_kv_cache(self, use_cache):
        if use_cache:
            print("Enabling input and KV cache!")
        else:
            print("Disabling input and KV cache!")

        self.use_input_cache = use_cache
        self.cache = self._init_cache()
        self.t5_decoder.reset_cache(use_cache=use_cache)

    @staticmethod
    def _init_cache():
        return {
            'hidden_states': None,
            'speech_mask': None,
            'input_audio_tokens': None,
            'target_text_tokens': None,
            'speech_encoder_emb': None,
            'char_embs': None,
        }
