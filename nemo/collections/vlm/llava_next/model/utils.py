# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

# 'These functions implementation is adapted from
# https://github.com/huggingface/transformers/blob/
# 53fad641cfdb5105e2470bcf3ef17ea8e25cc300/src/transformers/models/llava_next/modeling_llava_next.py'


def get_image_sequence_length(img_h, img_w, patch_dim, add_class_token, class_token_len):
    """Get image sequence length given image size, patch size, and class token."""
    num_patches_per_dim_h = img_h // patch_dim
    num_patches_per_dim_w = img_w // patch_dim
    num_patches = num_patches_per_dim_h * num_patches_per_dim_w
    return num_patches + (class_token_len if add_class_token else 0)


def merge_input_ids_with_image_features(
    image_features,
    feature_lens,
    inputs_embeds,
    input_ids,
    attention_mask,
    position_ids=None,
    labels=None,
    image_token_index=-200,
    ignore_index=-100,
):
    """
    Merge input_ids with with image features into final embeddings
    Args:
        image_features (`torch.Tensor` of shape `(all_feature_lens, embed_dim)`):
            All vision vectors of all images in the batch
        feature_lens (`torch.LongTensor` of shape `(num_images)`):
            The length of visual embeddings of each image as stacked in `image_features`
        inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, embed_dim)`):
            Token embeddings before merging with visual embeddings
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Input_ids of tokens, possibly filled with image token
        attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Mask to avoid performing attention on padding token indices.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.
        labels (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*)
            :abels need to be recalculated to support training (if provided)
        image_token_index (`int`, *optional*)
            Token id used to indicate the special "image" token. Defaults to `config.image_token_index`
        ignore_index (`int`, *optional*)
            Value that is used to pad `labels` and will be ignored when calculated loss. Default: -100.
    Returns:
        final_embedding, final_attention_mask, position_ids, final_labels
    Explanation:
        each image has variable length embeddings, with length specified by feature_lens
        image_features is concatenation of all visual embed vectors
        task: fill each <image> with the correct number of visual embeddings
        Example:
            X (5 patches), Y (3 patches), Z (8)
            X, Y are in the same sequence (in-context learning)
        if right padding
            input_ids: [
                a b c d e f X g h i j k Y l m
                o p q r Z s t u v _ _ _ _ _ _
            ]
            input_ids should be: [
                a b c d e f X X X X X g h i j k Y Y Y l m
                o p q r Z Z Z Z Z Z Z Z s t u v _ _ _ _ _
            ]
            labels should be: [
                a b c d e f _ _ _ _ _ g h i j k _ _ _ l m
                o p q r _ _ _ _ _ _ _ _ s t u v _ _ _ _ _
            ]
        elif left padding
            input_ids: [
                a b c d e f X g h i j k Y l m
                _ _ _ _ _ _ o p q r Z s t u v
            ]
            input_ids should be: [
                a b c d e f X X X X X g h i j k Y Y Y l m
                _ _ _ _ _ o p q r Z Z Z Z Z Z Z Z s t u v
            ]
            labels should be: [
                a b c d e f _ _ _ _ _ g h i j k _ _ _ l m
                _ _ _ _ _ o p q r _ _ _ _ _ _ _ _ s t u v
            ]
        Edge cases:
            * If tokens are same but image token sizes are different, then cannot infer left or right padding
            ```python
            cat_img = Image.open(requests.get(
                "http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)
            chart_img = Image.open(requests.get(
                "https://github.com/haotian-liu/LLaVA/blob/"
                "1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
                , stream=True).raw)
            prompts = [
                "[INST] <image>\nWhat is shown in this image? [/INST]",
                "[INST] <image>\nWhat is shown in this image? [/INST]",
            ]
            inputs = processor(prompts, [chart_img, cat_img], return_tensors='pt', padding=True).to("cuda")
                chart_img has 2634 tokens, while cat_img has 2340 tokens
            ```
            input_ids: [
                a b c d X g h
                i j Y k l m n
            ]
            where X is 3 tokens while Y is 5, this mean after merge
            if left-padding (batched generation)
                input_ids should be: [
                    _ _ a b c d X X X g h
                    i j Y Y Y Y Y k l m n
                ]
            elif (right padding) (training)
                input_ids should be: [
                    a b c d X X X g h _ _
                    i j Y Y Y Y Y k l m n
                ]
    """

    padding_side = 'right'
    pad_token_id = 0
    with torch.no_grad():
        # ! in llava 1.6, number of patches is variable
        num_images = feature_lens.size(0)
        num_image_features, embed_dim = image_features.shape
        if feature_lens.sum() != num_image_features:
            raise ValueError(f"{feature_lens=} / {feature_lens.sum()} != {image_features.shape=}")
        batch_size = input_ids.shape[0]
        _left_padding = torch.any(attention_mask[:, 0] == 0)
        _right_padding = torch.any(attention_mask[:, -1] == 0)

        left_padding = padding_side == "left"
        if batch_size > 1:
            if _left_padding and _right_padding:
                raise ValueError(f"both side of attention_mask has zero, invalid. {attention_mask}")
            elif _right_padding and left_padding:
                left_padding = False
            elif _left_padding and not left_padding:
                left_padding = True
        # Whether to turn off right padding
        # 1. Create a mask to know where special image tokens are
        special_image_token_mask = input_ids == image_token_index
        # special_image_token_mask: [bsz, seqlen]
        num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
        # num_special_image_tokens: [bsz]
        # Reserve for padding of num_images
        total_num_special_image_tokens = torch.sum(special_image_token_mask)
        if total_num_special_image_tokens != num_images:
            raise ValueError(
                f"Number of image tokens in input_ids ({total_num_special_image_tokens}) "
                f"different from num_images ({num_images})."
            )
        # Compute the maximum embed dimension
        # max_image_feature_lens is max_feature_lens per batch
        feature_lens = feature_lens.to(input_ids.device)
        feature_lens_batch = feature_lens.split(num_special_image_tokens.tolist(), dim=0)
        feature_lens_batch_sum = torch.tensor([x.sum() for x in feature_lens_batch], device=input_ids.device)
        embed_sequence_lengths = (
            (attention_mask == 1).long().sum(-1) - num_special_image_tokens + feature_lens_batch_sum
        )
        max_embed_dim = embed_sequence_lengths.max()
        batch_indices, non_image_indices = torch.where((input_ids != image_token_index) & (attention_mask == 1))
        # batch_indices, non_image_indices = torch.where((input_ids != image_token_index) )
        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged image-text sequence.
        # `special_image_token_mask` identifies image tokens.
        # Each image token will be replaced by `nb_text_tokens_per_images` text tokens.
        # `torch.cumsum` computes how each image token shifts subsequent text token positions.
        # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
        # ! instead of special_image_token_mask * (num_image_patches - 1)
        #   special_image_token_mask * (num_feature_len - 1)
        special_image_token_mask = special_image_token_mask.long()
        special_image_token_mask[special_image_token_mask == 1] = feature_lens - 1
        new_token_positions = torch.cumsum((special_image_token_mask + 1), -1) - 1
        if left_padding:
            # shift right token positions so that they are ending at the same number
            # the below here was incorrect?
            # new_token_positions += new_token_positions[:, -1].max() - new_token_positions[:, -1:]
            new_token_positions += max_embed_dim - 1 - new_token_positions[:, -1:]

        text_to_overwrite = new_token_positions[batch_indices, non_image_indices]
    # 3. Create the full embedding, already padded to the maximum position
    final_embedding = torch.zeros(
        batch_size, max_embed_dim, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
    )
    final_attention_mask = torch.zeros(
        batch_size, max_embed_dim, dtype=attention_mask.dtype, device=inputs_embeds.device
    )
    final_input_ids = torch.full(
        (batch_size, max_embed_dim), pad_token_id, dtype=input_ids.dtype, device=inputs_embeds.device
    )
    # In case the Vision model or the Language model has been offloaded to CPU, we need to manually
    # set the corresponding tensors into their correct target device.
    target_device = inputs_embeds.device
    batch_indices, non_image_indices, text_to_overwrite = (
        batch_indices.to(target_device),
        non_image_indices.to(target_device),
        text_to_overwrite.to(target_device),
    )
    attention_mask = attention_mask.to(target_device)
    input_ids = input_ids.to(target_device)

    # 4. Fill the embeddings based on the mask. If we have ["hey" "<image>", "how", "are"]
    # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the image features
    final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]
    final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_image_indices]
    final_input_ids[batch_indices, text_to_overwrite] = input_ids[batch_indices, non_image_indices]
    final_labels = None
    if labels is not None:
        labels = labels.to(target_device)
        final_labels = torch.full_like(final_attention_mask, ignore_index).to(torch.long)
        final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_image_indices]

    # 5. Fill the embeddings corresponding to the images. Anything that is not `text_positions` needs filling (#29835)
    with torch.no_grad():
        image_to_overwrite = torch.full(
            (batch_size, max_embed_dim), True, dtype=torch.bool, device=inputs_embeds.device
        )
        image_to_overwrite[batch_indices, text_to_overwrite] = False
        embed_indices = torch.arange(max_embed_dim).unsqueeze(0).to(target_device)
        embed_indices = embed_indices.expand(batch_size, max_embed_dim)
        embed_seq_lens = embed_sequence_lengths[:, None].to(target_device)

        if left_padding:
            # exclude padding on the left
            max_embed_dim = max_embed_dim.to(target_device)
            val = (max_embed_dim - embed_indices) <= embed_seq_lens
        else:
            # exclude padding on the right
            val = embed_indices < embed_seq_lens
        image_to_overwrite &= val

        if image_to_overwrite.sum() != num_image_features:
            raise ValueError(
                f"{image_to_overwrite.sum()=} != {num_image_features=} The input provided to the model are wrong. "
                f"The number of image tokens is {torch.sum(special_image_token_mask)} while"
                f" the number of image given to the model is {num_images}. "
                f"This prevents correct indexing and breaks batch generation."
            )
    final_embedding[image_to_overwrite] = image_features.contiguous().reshape(-1, embed_dim).to(target_device)
    final_attention_mask |= image_to_overwrite
    position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)
    final_loss_mask = None
    if final_labels is not None:
        final_loss_mask = (final_labels != ignore_index).long()

    return final_embedding, final_attention_mask, position_ids, final_labels, final_input_ids, final_loss_mask


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.
    Args:
        tensor (`torch.Tensor`):
            The image tensor, assumed to be of shape (num_channels, height, width).
        original_size (`tuple`):
            The original size of the image (height, width).
    Returns:
        `torch.Tensor`: The unpadded image tensor.
    """
    import numpy as np

    if not isinstance(original_size, (list, tuple)):
        if not isinstance(original_size, (torch.Tensor, np.ndarray)):
            raise TypeError(
                f"image_size invalid type: {type(original_size)} not valid ",
                "should be either list, tuple, np.ndarray or tensor",
            )
        original_size = original_size.tolist()
    original_height, original_width = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor


def select_best_resolution(original_size: tuple, possible_resolutions: list) -> tuple:
    """
    Selects the best resolution from a list of possible resolutions based on the original size.
    This is done by calculating the effective and wasted resolution for each possible resolution.
    The best fit resolution is the one that maximizes the effective resolution and minimizes the wasted resolution.
    Args:
        original_size (tuple):
            The original size of the image in the format (height, width).
        possible_resolutions (list):
            A list of possible resolutions in the format [(height1, width1), (height2, width2), ...].
    Returns:
        tuple: The best fit resolution in the format (height, width).
    """
    original_height, original_width = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for height, width in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (
            effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution
        ):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (height, width)

    return best_fit


def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):
    """
    Calculate the shape of the image patch grid after the preprocessing for images of any resolution.
    Args:
        image_size (`tuple`):
            The size of the input image in the format (width, height).
        grid_pinpoints (`List`):
            A list containing possible resolutions. Each item in the list should be a tuple or list
            of the form `(height, width)`.
        patch_size (`int`):
            The size of each image patch.
    Returns:
        tuple: The shape of the image patch grid in the format (width, height).
    """
    import numpy as np

    if not isinstance(grid_pinpoints, list):
        raise TypeError("grid_pinpoints should be a list of tuples or lists")

    # ! VERY IMPORTANT if image_size is tensor, must convert to into tuple, otherwise it will cause wrong calculate
    if not isinstance(image_size, (list, tuple)):
        if not isinstance(image_size, (torch.Tensor, np.ndarray)):
            raise TypeError(
                f"image_size invalid type: {type(image_size)} not valid, "
                "should be either list, tuple, np.ndarray or tensor"
            )
        image_size = image_size.tolist()

    height, width = select_best_resolution(image_size, grid_pinpoints)
    return height // patch_size, width // patch_size


# These functions implementation is adapted from
# https://github.com/huggingface/transformers/blob/
# 53fad641cfdb5105e2470bcf3ef17ea8e25cc300/src/transformers/models/llava_next/modeling_llava_next.py#L655'


def pack_image_features(image_features, image_sizes, vision_feature_select_strategy, image_newline=None):
    """
    Reshape, unpad and then pack each image_feature into a single image_features tensor containing all visual vectors.
    Args:
        image_features (`List[torch.Tensor]` of length num_images,
        each of shape `(num_patches, image_length, embed_dim)`)
            List of image feature tensor, each contains all the visual feature of all patches.
        image_sizes (`torch.Tensor` of shape `(num_images, 2)`)
            Actual image size of each images (H, W).
        vision_feature_select_strategy (`str`)
            The feature selection strategy used to select the vision feature from the vision backbone.
        image_newline (`torch.Tensor` of shape `(embed_dim)`)
            New line embedding vector.
    Returns:
        image_features (`torch.Tensor` of shape `(all_feat_len, embed_dim)`)
        feature_lens (`List[int]`)
            token length of each image in image_features
    """
    from transformers import LlavaNextConfig

    config = LlavaNextConfig()
    new_image_features = []
    feature_lens = []

    for image_idx, image_feature in enumerate(image_features):
        if image_feature.shape[0] > 1:
            base_image_feature = image_feature[0]
            image_feature = image_feature[1:]
            height = width = config.vision_config.image_size // config.vision_config.patch_size

            if vision_feature_select_strategy == "default":
                expected_num_patches = height * width
            elif vision_feature_select_strategy == "full":
                expected_num_patches = height * width + 1
            if expected_num_patches != base_image_feature.shape[0]:
                raise ValueError("The number of patches is not consistent with the image size.")

            num_patch_height, num_patch_width = get_anyres_image_grid_shape(
                image_sizes[image_idx],
                config.image_grid_pinpoints,
                config.vision_config.image_size,
            )
            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
            image_feature = unpad_image(image_feature, image_sizes[image_idx])
            if image_newline is not None:
                image_feature = torch.cat(
                    (
                        image_feature,
                        image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.dtype),
                    ),
                    dim=-1,
                )
            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
            image_feature = torch.cat((base_image_feature, image_feature), dim=0)
        else:
            image_feature = image_feature[0]
            if image_newline is not None:
                image_feature = torch.cat((image_feature, image_newline[None].to(image_feature)), dim=0)
        new_image_features.append(image_feature)
        feature_lens.append(image_feature.size(0))
    image_features = torch.cat(new_image_features, dim=0)
    feature_lens = torch.tensor(feature_lens, dtype=torch.long, device=image_features.device)
    return image_features, feature_lens
