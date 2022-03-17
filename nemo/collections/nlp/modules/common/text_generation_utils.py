# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

"""Utilities for generating text."""

import torch
import torch.nn.functional as F

from nemo.collections.common.tokenizers.tabular_tokenizer import TabularTokenizer
from nemo.collections.nlp.modules.common.megatron.utils import get_ltor_masks_and_position_ids
from nemo.utils import AppState

try:
    from apex.transformer import parallel_state
    from apex.transformer.pipeline_parallel.schedules.fwd_bwd_pipelining_without_interleaving import (
        forward_backward_pipelining_without_interleaving,
    )
    from apex.transformer.pipeline_parallel.schedules.fwd_bwd_no_pipelining import forward_backward_no_pipelining
    from apex.transformer.pipeline_parallel.utils import _reconfigure_microbatch_calculator

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False


def get_batch(model, tokenizer, context_tokens):
    """Generate batch from context tokens."""
    # Move to GPU.
    tokens = context_tokens.contiguous().cuda()
    # Get the attention mask and postition ids.
    attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eos_id,
        model.cfg.get('reset_position_ids', False),
        model.cfg.get('reset_attention_mask', False),
        model.cfg.get('eod_mask_loss', False),
    )

    return tokens, attention_mask, position_ids


def tab_logits(logits, min_id, max_id, filter_value=-float('Inf')):
    logits[:, :min_id] = filter_value
    logits[:, max_id:] = filter_value
    return logits


def top_k_logits(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ This function has been mostly taken from huggingface conversational
     ai code at
         https://medium.com/huggingface/how-to-build-a-state-of-the-art-
              conversational-ai-with-transfer-learning-2d818ac26313 """

    if top_k > 0:
        # Remove all tokens with a probability less than the
        # last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Cconvert to 1D
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token
        # above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        for i in range(sorted_indices.size(0)):
            indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
            logits[i][indices_to_remove] = filter_value

    return logits


def pad_batch(batch, pad_id, max_len):
    context_lengths = []
    max_context_length = max([len(tokens) for tokens in batch])
    for tokens in batch:
        context_length = len(tokens)
        if context_length < max_context_length + max_len:
            tokens.extend([pad_id] * (max_context_length + max_len - context_length))
        context_lengths.append(context_length)
    return batch, context_lengths


def tokenize_batch(tokenizer, sentences, max_len, add_BOS):
    if add_BOS:
        context_tokens = [[tokenizer.eos_id] + tokenizer.text_to_ids(s) for s in sentences]
    else:
        context_tokens = [tokenizer.text_to_ids(s) for s in sentences]
    context_tokens, context_lengths = pad_batch(context_tokens, tokenizer.eos_id, max_len)
    context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
    context_length_tensor = torch.cuda.LongTensor(context_lengths)
    return context_tokens_tensor, context_length_tensor


def send_generate_info(
    context_tokens_tensor, context_length_tensor, tokens_to_generate, all_probs, temperature, top_k, top_p, greedy
):
    """
    Needs to be synced up with receive_generate_info
    """
    # Send the sizes of the tensors
    input_info = [
        context_tokens_tensor.size(0),  # batch_size
        context_tokens_tensor.size(1),  # seq_len
        tokens_to_generate,
        all_probs,
        temperature,
        top_k,
        top_p,
        greedy,
    ]
    input_info_tensor = torch.cuda.FloatTensor(input_info)
    torch.distributed.broadcast(input_info_tensor, 0)

    # Send variables to all ranks
    torch.distributed.broadcast(context_length_tensor, 0)
    torch.distributed.broadcast(context_tokens_tensor, 0)


def receive_generate_info():
    """
    Needs to be synced up with send_generate_info
    """
    input_info_tensor = torch.empty(8, dtype=torch.float32, device=torch.cuda.current_device())
    torch.distributed.broadcast(input_info_tensor, 0)
    batch_size = int(input_info_tensor[0].item())
    seq_len = int(input_info_tensor[1].item())
    tokens_to_generate = int(input_info_tensor[2].item())
    all_probs = bool(input_info_tensor[3].item())
    temperature = float(input_info_tensor[4].item())
    top_k = float(input_info_tensor[5].item())
    top_p = float(input_info_tensor[6].item())
    greedy = bool(input_info_tensor[7].item())

    context_length_tensor = torch.empty(batch_size, dtype=torch.int64, device=torch.cuda.current_device())
    context_tokens_tensor = torch.empty(batch_size, seq_len, dtype=torch.int64, device=torch.cuda.current_device())

    # Send variables to all ranks
    torch.distributed.broadcast(context_length_tensor, 0)
    torch.distributed.broadcast(context_tokens_tensor, 0)

    return (
        context_length_tensor,
        context_tokens_tensor,
        tokens_to_generate,
        all_probs,
        temperature,
        top_k,
        top_p,
        greedy,
    )


def synced_generate(
    model,
    context_tokens_tensor,
    context_length_tensor,
    tokens_to_generate,
    all_probs,
    temperature,
    top_k=0,
    top_p=0.0,
    greedy=False,
):
    context_length = context_length_tensor.min().item()
    tokenizer = model.tokenizer
    tokens, attention_mask, position_ids = get_batch(model, tokenizer, context_tokens_tensor)
    if isinstance(tokenizer, TabularTokenizer):
        batch_token_iterator = tab_sample_sequence_batch(
            model,
            context_tokens_tensor,
            context_length_tensor,
            attention_mask,
            position_ids,
            tokens_to_generate,
            all_probs,
            temperature=temperature,
        )
    else:
        batch_token_iterator = sample_sequence_batch(
            model,
            context_tokens_tensor,
            context_length_tensor,
            attention_mask,
            position_ids,
            tokens_to_generate,
            all_probs,
            temperature=temperature,
            extra={"top_p": top_p, "top_k": top_k, "greedy": greedy},
        )

    for tokens, lengths, output_logits, full_logits in batch_token_iterator:
        context_length += 1

    if parallel_state.is_pipeline_last_stage():
        src = parallel_state.get_pipeline_model_parallel_last_rank()
        group = parallel_state.get_embedding_group()
        torch.distributed.broadcast(output_logits, src, group)
        if all_probs:
            src = parallel_state.get_pipeline_model_parallel_last_rank()
            group = parallel_state.get_embedding_group()
            torch.distributed.broadcast(full_logits, src, group)

    else:
        if parallel_state.is_pipeline_first_stage():
            src = parallel_state.get_pipeline_model_parallel_last_rank()
            group = parallel_state.get_embedding_group()
            output_logits = torch.empty(
                tokens.size(0), context_length - 1, dtype=torch.float32, device=torch.device("cuda")
            )
            torch.distributed.broadcast(output_logits, src, group)

            if all_probs:
                src = parallel_state.get_pipeline_model_parallel_last_rank()
                group = parallel_state.get_embedding_group()
                full_logits = torch.empty(
                    tokens.size(0),
                    context_length,
                    model.padded_vocab_size,
                    dtype=torch.float32,
                    device=torch.device("cuda"),
                )
                torch.distributed.broadcast(full_logits, src, group)
    if tokens is not None:
        return tokens[:, :context_length], output_logits, full_logits


def generate(
    model,
    sentences=None,
    tokens_to_generate=0,
    all_probs=False,
    temperature=1.0,
    add_BOS=False,
    top_k=0,
    top_p=0.0,
    greedy=False,
):
    model.eval()
    tokenizer = model.tokenizer
    if torch.distributed.get_rank() == 0:
        context_tokens_tensor, context_length_tensor = tokenize_batch(
            tokenizer, sentences, tokens_to_generate, add_BOS
        )
        send_generate_info(
            context_tokens_tensor,
            context_length_tensor,
            tokens_to_generate,
            all_probs,
            temperature,
            top_k,
            top_p,
            greedy,
        )
    else:
        (
            context_length_tensor,
            context_tokens_tensor,
            tokens_to_generate,
            all_probs,
            temperature,
            top_k,
            top_p,
            greedy,
        ) = receive_generate_info()

    output = synced_generate(
        model,
        context_tokens_tensor,
        context_length_tensor,
        tokens_to_generate,
        all_probs,
        temperature,
        top_k=top_k,
        top_p=top_p,
        greedy=greedy,
    )
    if output is not None:
        decode_tokens, output_logits, full_logits = output
        resp_sentences = []
        resp_sentences_seg = []

        decode_tokens = decode_tokens.cpu().numpy().tolist()
        for decode_token in decode_tokens:
            sentence = tokenizer.ids_to_text(decode_token)
            resp_sentences.append(sentence)
            if not isinstance(tokenizer, TabularTokenizer):
                words = []
                for token in decode_token:
                    word = tokenizer.tokenizer.decoder[token]
                    word = bytearray([tokenizer.tokenizer.byte_decoder[c] for c in word]).decode(
                        'utf-8', errors='replace'
                    )
                    words.append(word)
                resp_sentences_seg.append(words)
            else:
                words = tokenizer.text_to_tokens(sentence)
                resp_sentences_seg.append(words)
        output_logits = output_logits.cpu().numpy().tolist()
        if all_probs:
            full_logits = full_logits.cpu().numpy().tolist()

        return resp_sentences, resp_sentences_seg, output_logits, full_logits, decode_tokens


def switch(val1, val2, boolean):
    boolean = boolean.type_as(val1)
    return (1 - boolean) * val1 + boolean * val2


def forward_step(model, batch, tensor_shape):

    if model.cfg.get('pipeline_model_parallel_size', 1) > 1:
        output_tensor = forward_backward_pipelining_without_interleaving(
            forward_step_func=model.get_forward_output_only_func(),
            batch=batch,
            model=model.model,
            forward_only=True,
            tensor_shape=tensor_shape,
            dtype=model.autocast_dtype,
        )
    else:
        output_tensor = forward_backward_no_pipelining(
            forward_step_func=model.get_forward_output_only_func(),
            batch=batch,
            model=model.model,
            forward_only=True,
            tensor_shape=tensor_shape,
            dtype=model.autocast_dtype,
        )
    return output_tensor


def sample_sequence_batch(
    model,
    context_tokens,
    context_lengths,
    attention_mask,
    position_ids,
    tokens_to_generate,
    all_probs=False,
    type_ids=None,
    temperature=None,
    extra={},
):
    app_state = AppState()
    micro_batch_size = context_tokens.shape[0]
    _reconfigure_microbatch_calculator(
        rank=app_state.global_rank,
        rampup_batch_size=None,
        global_batch_size=micro_batch_size,
        micro_batch_size=micro_batch_size,
        data_parallel_size=1,
    )
    tokenizer = model.tokenizer
    model.eval()
    with torch.no_grad():
        context_length = context_lengths.min().item()

        # added eos_id to support the function generate_samples_eval that passes
        # eos_id as an argument and needs termination when that id id found.
        eod_id = tokenizer.eos_id
        counter = 0

        batch_size = context_tokens.size(0)
        is_done = torch.zeros([batch_size]).byte().cuda()
        tokens = context_tokens
        output_logits = None

        # Generate enough tokens for the longest sequence
        maxlen = tokens_to_generate + context_lengths.max().item()

        if maxlen > model.cfg.encoder_seq_length:
            maxlen = model.cfg.encoder_seq_length

        lengths = torch.ones([batch_size]).long().cuda() * maxlen

        while context_length < maxlen:
            types2use = None
            if counter == 0:
                # Allocate memory for the entire context.
                set_inference_key_value_memory = True
                tokens2use = tokens[:, :context_length]
                positions2use = position_ids[:, :context_length]
                if type_ids is not None:
                    types2use = type_ids[:, :context_length]
            else:
                # Set this to false so the memory is not reallocated.
                set_inference_key_value_memory = False
                tokens2use = tokens[:, context_length - 1].view(batch_size, -1)
                positions2use = position_ids[:, context_length - 1].view(batch_size, -1)
                if type_ids is not None:
                    types2use = type_ids[:, context_length - 1].view(batch_size, -1)

            attention_mask_repeat = torch.concat([attention_mask for _ in range(micro_batch_size)])
            setkey_value_array = torch.tensor(
                [set_inference_key_value_memory] * micro_batch_size, device=torch.cuda.current_device()
            )
            len_array = torch.tensor([maxlen] * micro_batch_size, device=torch.cuda.current_device())
            batch = [tokens2use, attention_mask_repeat, positions2use, setkey_value_array, len_array]
            tensor_shape = [tokens2use.shape[1], micro_batch_size, model.cfg.hidden_size]

            output = forward_step(model, batch, tensor_shape)

            if parallel_state.is_pipeline_last_stage():
                output = output[0]['logits'].float()
                assert output is not None
                output = output.float()
                logits = output[:, -1].view(batch_size, -1).contiguous()

                if extra.get('greedy', False):  # args.greedy:
                    prev = torch.argmax(logits, dim=-1).view(-1)
                else:
                    logits = logits.float()
                    logits /= temperature
                    logits = top_k_logits(logits, top_k=extra.get('top_k', 0), top_p=extra.get('top_p', 0.9))
                    log_probs = F.softmax(logits, dim=-1)
                    prev = torch.multinomial(log_probs, num_samples=1).view(-1)
                started = context_lengths <= context_length

                # Clamp the out of vocabulary tokens.
                prev = torch.clamp(prev, max=tokenizer.vocab_size - 1)

                new_tokens = switch(tokens[:, context_length].view(-1), prev, started)
                tokens[:, context_length] = new_tokens

                if output_logits is None:
                    output_context = F.log_softmax(output[:, :context_length, :], 2)
                    indices = torch.unsqueeze(tokens[:, 1 : context_length + 1], 2)
                    output_logits = torch.gather(output_context, 2, indices).squeeze(2)
                    if all_probs:
                        full_logits = output_context
                else:
                    output_context = F.log_softmax(output, 2)
                    indices = torch.unsqueeze(new_tokens, 1).unsqueeze(2)
                    new_output_logits = torch.gather(output_context, 2, indices).squeeze(2)

                    # TODO(rprenger) we're copying output_logits every time.  Should pre-allocate
                    output_logits = torch.cat([output_logits, new_output_logits], 1)
                    if all_probs:
                        full_logits = torch.cat([full_logits, output_context], 1)

                src = parallel_state.get_pipeline_model_parallel_last_rank()
                group = parallel_state.get_embedding_group()
                torch.distributed.broadcast(new_tokens, src, group)

                done_token = (prev == eod_id).byte() & started.byte()
                just_finished = (done_token & ~is_done).bool()
                lengths[just_finished.view(-1)] = context_length
                is_done = is_done | done_token

                done = torch.all(is_done)
                src = parallel_state.get_pipeline_model_parallel_last_rank()
                group = parallel_state.get_pipeline_model_parallel_group()
                torch.distributed.broadcast(done, src, group)
                if all_probs:
                    yield tokens, lengths, output_logits, full_logits
                else:
                    yield tokens, lengths, output_logits, None

            else:
                if parallel_state.is_pipeline_first_stage():
                    src = parallel_state.get_pipeline_model_parallel_last_rank()
                    group = parallel_state.get_embedding_group()
                    new_tokens = torch.empty_like(tokens[:, context_length])
                    torch.distributed.broadcast(new_tokens, src, group)
                    tokens[:, context_length] = new_tokens
                    yield tokens, None, None, None
                else:
                    yield None, None, None, None

                done = torch.cuda.ByteTensor([0])
                src = parallel_state.get_pipeline_model_parallel_last_rank()
                group = parallel_state.get_pipeline_model_parallel_group()
                torch.distributed.broadcast(done, src, group)

            context_length += 1
            counter += 1
            if done:
                break


def tab_sample_sequence_batch(
    model,
    context_tokens,
    context_lengths,
    attention_mask,
    position_ids,
    tokens_to_generate,
    all_probs=False,
    type_ids=None,
    temperature=None,
):
    app_state = AppState()
    micro_batch_size = context_tokens.shape[0]
    _reconfigure_microbatch_calculator(
        rank=app_state.global_rank,
        rampup_batch_size=None,
        global_batch_size=micro_batch_size,
        micro_batch_size=micro_batch_size,
        data_parallel_size=1,
    )
    tokenizer = model.tokenizer
    sizes = tokenizer.code_column.sizes
    tokens_per_row = sum(sizes) + 1
    columns = tokenizer.code_column.columns
    num_columns = len(columns)
    tokenid_range = []
    for i in range(num_columns):
        tokenid_range.extend(tokenizer.code_column.get_range(i))

    model.eval()
    with torch.no_grad():
        context_length = context_lengths.min().item()
        context = context_tokens[:, :context_length]
        # the context may start in the middle of the row,
        # calculate the offset according to the position of '\n' or '<|endoftext|>'
        positions = torch.where(context == tokenizer.eor)[1]
        if len(positions) == 0:
            positions = torch.where(context == tokenizer.eod)[1]
        if len(positions) != 0:
            max_position = positions.max().item()
            # TODO, need to make sure context of different batch have the same offset lengths")
            # otherwise, need to calculate offset per batch_id
            offset = (context_length - max_position - 1) % tokens_per_row
        else:
            offset = 0

        eod_id = tokenizer.eos_id

        counter = 0

        batch_size = context_tokens.size(0)
        is_done = torch.zeros([batch_size]).byte().cuda()
        tokens = context_tokens
        output_logits = None

        # Generate enough tokens for the longest sequence
        maxlen = tokens_to_generate + context_lengths.max().item()

        if maxlen > model.cfg.encoder_seq_length:
            maxlen = model.cfg.encoder_seq_length

        lengths = torch.ones([batch_size]).long().cuda() * maxlen

        while context_length < maxlen:
            types2use = None
            if counter == 0:
                # Allocate memory for the entire context.
                set_inference_key_value_memory = True
                tokens2use = tokens[:, :context_length]
                positions2use = position_ids[:, :context_length]
                if type_ids is not None:
                    types2use = type_ids[:, :context_length]
            else:
                # Set this to false so the memory is not reallocated.
                set_inference_key_value_memory = False
                tokens2use = tokens[:, context_length - 1].view(batch_size, -1)
                positions2use = position_ids[:, context_length - 1].view(batch_size, -1)
                if type_ids is not None:
                    types2use = type_ids[:, context_length - 1].view(batch_size, -1)
            # micro_batch_size = 2
            attention_mask_repeat = torch.concat([attention_mask for _ in range(micro_batch_size)])
            setkey_value_array = torch.tensor(
                [set_inference_key_value_memory] * micro_batch_size, device=torch.cuda.current_device()
            )
            len_array = torch.tensor([maxlen] * micro_batch_size, device=torch.cuda.current_device())
            batch = [tokens2use, attention_mask_repeat, positions2use, setkey_value_array, len_array]
            tensor_shape = [tokens2use.shape[1], micro_batch_size, model.cfg.hidden_size]

            output = forward_step(model, batch, tensor_shape)

            if parallel_state.is_pipeline_last_stage():
                output = output[0]['logits'].float()
                assert output is not None
                output = output.float()
                logits = output[:, -1].view(batch_size, -1).contiguous()
                token_in_row = (counter + offset) % tokens_per_row
                if False:  # args.greedy:
                    prev = torch.argmax(logits, dim=-1).view(-1)
                else:
                    logits = logits.float()
                    logits /= temperature
                    if token_in_row == tokens_per_row - 1:
                        # line break
                        eor_id = tokenizer.eor
                        eod_id = tokenizer.eos_id
                        min_id = min(eor_id, eod_id)
                        max_id = max(eor_id, eod_id) + 1
                        logits = tab_logits(logits, min_id, max_id)
                    else:
                        # limit the range
                        min_id, max_id = tokenid_range[token_in_row]
                        logits = tab_logits(logits, min_id, max_id)
                    log_probs = F.softmax(logits, dim=-1)
                    prev = torch.multinomial(log_probs, num_samples=1).view(-1)
                    # simulate the eos_id
                    # if counter == 59:
                    #     prev[:] = eos_id
                started = context_lengths <= context_length

                # Clamp the out of vocabulary tokens.
                prev = torch.clamp(prev, max=tokenizer.vocab_size - 1)

                new_tokens = switch(tokens[:, context_length].view(-1), prev, started)
                tokens[:, context_length] = new_tokens

                if output_logits is None:
                    output_context = F.log_softmax(output[:, :context_length, :], 2)
                    indices = torch.unsqueeze(tokens[:, 1 : context_length + 1], 2)
                    output_logits = torch.gather(output_context, 2, indices).squeeze(2)
                    if all_probs:
                        full_logits = output_context
                else:
                    output_context = F.log_softmax(output, 2)
                    indices = torch.unsqueeze(new_tokens, 1).unsqueeze(2)
                    new_output_logits = torch.gather(output_context, 2, indices).squeeze(2)

                    # TODO(rprenger) we're copying output_logits every time.  Should pre-allocate
                    output_logits = torch.cat([output_logits, new_output_logits], 1)
                    if all_probs:
                        full_logits = torch.cat([full_logits, output_context], 1)

                src = parallel_state.get_pipeline_model_parallel_last_rank()
                group = parallel_state.get_embedding_group()
                torch.distributed.broadcast(new_tokens, src, group)

                done_token = (prev == eod_id).byte() & started.byte()
                just_finished = (done_token & ~is_done).bool()
                lengths[just_finished.view(-1)] = context_length
                is_done = is_done | done_token

                done = torch.all(is_done)
                src = parallel_state.get_pipeline_model_parallel_last_rank()
                group = parallel_state.get_pipeline_model_parallel_group()
                torch.distributed.broadcast(done, src, group)
                if all_probs:
                    yield tokens, lengths, output_logits, full_logits
                else:
                    yield tokens, lengths, output_logits, None

            else:
                if parallel_state.is_pipeline_first_stage():
                    src = parallel_state.get_pipeline_model_parallel_last_rank()
                    group = parallel_state.get_embedding_group()
                    new_tokens = torch.empty_like(tokens[:, context_length])
                    torch.distributed.broadcast(new_tokens, src, group)
                    tokens[:, context_length] = new_tokens
                    yield tokens, None, None, None
                else:
                    yield None, None, None, None

                done = torch.cuda.ByteTensor([0])
                src = parallel_state.get_pipeline_model_parallel_last_rank()
                group = parallel_state.get_pipeline_model_parallel_group()
                torch.distributed.broadcast(done, src, group)

            context_length += 1
            counter += 1
            if done:
                break
