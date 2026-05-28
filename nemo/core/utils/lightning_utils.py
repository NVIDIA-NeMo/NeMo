# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""Helpers for working with PyTorch Lightning's ``training_step``."""
from typing import Any, Iterator, Tuple

import lightning.pytorch as pl


def read_batch(dataloader_iter: Iterator, model: pl.LightningModule) -> Tuple[Any, int]:
    """Pull the next batch from a Lightning ``dataloader_iter`` and apply the
    device/precision conversions that ``_PrefetchDataFetcher`` would have
    applied for the default ``training_step(batch, batch_idx)`` signature.

    Use this from a ``training_step(self, dataloader_iter)``-style step. That
    signature makes Lightning select ``_DataLoaderIterDataFetcher`` (no
    prefetch), which is required for bit-identical checkpoint resumption with
    a stateful dataloader: the default ``_PrefetchDataFetcher`` re-primes one
    batch on every iter init (including on resume), advancing the stateful
    dataloader past the saved snapshot point and giving the resumed run a
    one-batch drift versus the continuous run.

    Also force-fires StatelessTimer's preempt check before pulling the next
    batch. Under ``dataloader_iter`` flavor, Lightning's on_train_batch_end
    callback dispatch silently fails to invoke StatelessTimer (likely related
    to the "unforeseen effects on callbacks" warning Lightning emits for this
    flavor) — so the SLURM walltime preempt-save + graceful-exit path never
    fires, and each chunk's mid-epoch progress is lost to SIGKILL. Calling
    ``StatelessTimer._check_time_remaining`` directly here restores the same
    behavior the ``(batch, batch_idx)`` flavor has out of the box.

    Args:
        dataloader_iter: The iterator passed by Lightning into a
            ``training_step(self, dataloader_iter)`` (an instance of
            ``_DataFetcherWrapper``). Yields ``(batch, batch_idx, dataloader_idx)``.
        model: The ``LightningModule`` whose ``trainer`` carries the precision
            plugin and strategy used to move the batch to device.

    Returns:
        ``(batch, batch_idx)`` — batch is already converted to the right
        precision and moved to the model's device, ready for forward.
    """
    trainer = model.trainer
    _force_fire_stateless_timer(trainer)
    batch, batch_idx, dataloader_idx = next(dataloader_iter)
    batch = trainer.precision_plugin.convert_input(batch)
    batch = model._on_before_batch_transfer(batch, dataloader_idx=dataloader_idx)
    batch = trainer.strategy.batch_to_device(batch, dataloader_idx=dataloader_idx)
    return batch, batch_idx


def _force_fire_stateless_timer(trainer: pl.Trainer) -> None:
    """Invoke ``StatelessTimer._check_time_remaining`` directly.

    Workaround for Lightning's ``dataloader_iter`` step flavor: in that mode,
    StatelessTimer's ``on_train_batch_end`` does not reliably fire, so neither
    the time-elapsed check nor the preempt-save+exit path runs. Calling
    ``_check_time_remaining`` from inside the user-owned training_step makes
    the preempt behavior identical to what ``(batch, batch_idx)`` flavor
    gets via Lightning's standard callback flow.

    Idempotent on the time-not-yet-up case (cheap: one ``time_elapsed()``
    check + one comparison). On the time-up case, ``StatelessTimer`` saves a
    ``-last.ckpt`` via ``NeMoModelCheckpoint._save_last_checkpoint`` and
    raises ``_TunerExitException`` to exit Lightning gracefully — that
    exception propagates up through ``read_batch`` → ``training_step`` →
    Lightning's epoch loop, which Lightning treats as a clean stop.
    """
    # Local import to avoid a circular import at module load time
    # (exp_manager imports from various nemo submodules).
    from nemo.utils.exp_manager import StatelessTimer

    for cb in trainer.callbacks:
        if isinstance(cb, StatelessTimer):
            cb._check_time_remaining(trainer)
            return
