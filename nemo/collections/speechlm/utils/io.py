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

from pathlib import Path
from typing import Optional

import lightning.pytorch as pl

from nemo.lightning.io.mixin import ConnectorMixin, ModelConnector


def import_ckpt(
    model: pl.LightningModule,
    source: str,
    output_path: Optional[Path] = None,
    overwrite: bool = False,
    on_import_ckpt: bool = True,
) -> Path:
    """
    Overriding nemo/lightning/io/api.py:import_ckpt to add on_import_ckpt flag, which is used to

    Imports a checkpoint into a model using the model's associated importer, typically for
    the purpose of fine-tuning a community model trained in an external framework, such as
    Hugging Face. This function leverages the ConnectorMixin interface to integrate external
    checkpoint data seamlessly into the specified model instance.

    The importer component of the model reads the checkpoint data from the specified source
    and transforms it into the right format. This is particularly useful for adapting
    models that have been pre-trained in different environments or frameworks to be fine-tuned
    or further developed within the current system. The function allows for specifying an output
    path for the imported checkpoint; if not provided, the importer's default path will be used.
    The 'overwrite' parameter enables the replacement of existing data at the output path, which
    is useful when updating models with new data and discarding old checkpoint files.

    For instance, using `import_ckpt(Mistral7BModel(), "hf")` initiates the import process
    by searching for a registered model importer tagged with "hf". In NeMo, `HFMistral7BImporter`
    is registered under this tag via:
    `@io.model_importer(Mistral7BModel, "hf", default_path="mistralai/Mistral-7B-v0.1")`.
    This links `Mistral7BModel` to `HFMistral7BImporter`, designed for HuggingFace checkpoints.
    The importer then processes and integrates these checkpoints into `Mistral7BModel` for further
    fine-tuning.

    Args:
        model (pl.LightningModule): The model into which the checkpoint will be imported.
            This model must implement the ConnectorMixin, which includes the necessary
            importer method for checkpoint integration.
        source (str): The source from which the checkpoint will be imported. This can be
            a file path, URL, or any other string identifier that the model's importer
            can recognize.
        output_path (Optional[Path]): The path where the imported checkpoint will be stored.
            If not specified, the importer's default path is used.
        overwrite (bool): If set to True, existing files at the output path will be overwritten.
            This is useful for model updates where retaining old checkpoint files is not required.
        on_import_ckpt (bool): If set to True, the importer will also call importer.on_import_ckpt(model),
            which imports the tokenizer associated with the model. This is set to False for multi-modal LLMs
            like SpeechLM, where the tokenizer is associated with the SpeechLM model instead of the internal LLM.

    Returns
    -------
        Path: The path where the checkpoint has been saved after import. This path is determined
            by the importer, based on the provided output_path and its internal logic.

    Raises
    ------
        ValueError: If the model does not implement ConnectorMixin, indicating a lack of
            necessary importer functionality.

    Example:
        model = Mistral7BModel()
        imported_path = import_ckpt(model, "hf://mistralai/Mistral-7B-v0.1")
    """
    if not isinstance(model, ConnectorMixin):
        raise ValueError("Model must be an instance of ConnectorMixin")

    importer: ModelConnector = model.importer(source)
    ckpt_path = importer(overwrite=overwrite, output_path=output_path)
    if on_import_ckpt:
        importer.on_import_ckpt(model)
    return ckpt_path
