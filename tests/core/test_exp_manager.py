# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
import math
import re
from pathlib import Path

import pytest
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

from nemo.constants import NEMO_ENV_VARNAME_VERSION
from nemo.core.classes import ModelPT
from nemo.utils.exp_manager import (
    CheckpointMisconfigurationError,
    LoggerMisconfigurationError,
    NotFoundError,
    exp_manager,
)


class MyTestOptimizer(torch.optim.Optimizer):
    def __init__(self, params):
        self._step = 0
        super().__init__(params, {})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if self._step == 0:
                    p.data = 0.1 * torch.ones(p.shape)
                elif self._step == 1:
                    p.data = 0.0 * torch.ones(p.shape)
                else:
                    p.data = 0.01 * torch.ones(p.shape)
        self._step += 1
        return loss


class DoNothingOptimizer(torch.optim.Optimizer):
    def __init__(self, params):
        self._step = 0
        super().__init__(params, {})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._step += 1
        return loss


class OnesDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_len):
        super().__init__()
        self.__dataset_len = dataset_len

    def __getitem__(self, *args):
        return torch.ones(2)

    def __len__(self):
        return self.__dataset_len


class ExampleModel(ModelPT):
    def __init__(self, *args, **kwargs):
        cfg = OmegaConf.structured({})
        super().__init__(cfg)
        pl.seed_everything(1234)
        self.l1 = torch.nn.modules.Linear(in_features=2, out_features=1)

    def train_dataloader(self):
        dataset = OnesDataset(2)
        return torch.utils.data.DataLoader(dataset, batch_size=2)

    def val_dataloader(self):
        dataset = OnesDataset(10)
        return torch.utils.data.DataLoader(dataset, batch_size=2)

    def forward(self, batch):
        output = self.l1(batch)
        output = torch.nn.functional.l1_loss(output, torch.zeros(output.size()).to(output.device))
        return output

    def validation_step(self, batch, batch_idx):
        return self(batch)

    def training_step(self, batch, batch_idx):
        return self(batch)

    def configure_optimizers(self):
        return MyTestOptimizer(self.parameters())
        # return torch.optim.Adam(self.parameters(), lr=0.1)

    def list_available_models(self):
        pass

    def setup_training_data(self):
        pass

    def setup_validation_data(self):
        pass

    def validation_epoch_end(self, loss):
        self.log("val_loss", torch.stack(loss).mean())


class DoNothingModel(ExampleModel):
    def configure_optimizers(self):
        return DoNothingOptimizer(self.parameters())


class TestExpManager:
    @pytest.mark.unit
    def test_omegaconf(self):
        """Ensure omegaconf raises an error when an unexcepted argument is passed"""
        with pytest.raises(OmegaConfBaseException):
            exp_manager(pl.Trainer(accelerator='cpu'), {"unused": 1})

    @pytest.mark.unit
    def test_trainer_loggers(self, tmp_path):
        """ Test that a trainer with logger errors out with a number of arguments. Test that it works with
        create_tensorboard_logger set to False
        """
        test_trainer = pl.Trainer(accelerator='cpu')  # Should create logger and modelcheckpoint

        with pytest.raises(LoggerMisconfigurationError):  # Fails because exp_manager defaults to trainer
            exp_manager(test_trainer, {"exp_dir": str(tmp_path)})
        with pytest.raises(LoggerMisconfigurationError):  # Fails because exp_manager defaults to trainer
            exp_manager(test_trainer, {"explicit_log_dir": str(tmp_path)})
        with pytest.raises(LoggerMisconfigurationError):  # Fails because exp_manager defaults to trainer
            exp_manager(test_trainer, {"resume_if_exists": True})

        # Check that exp_manager uses trainer.logger, it's exp_dir, name, and version
        log_dir = exp_manager(test_trainer, {"create_tensorboard_logger": False, "create_checkpoint_callback": False})
        assert log_dir.resolve() == Path("./lightning_logs/version_0").resolve()
        assert Path("./lightning_logs").exists()
        assert Path("./lightning_logs/version_0").exists()

        # Check that a trainer without a logger gets a logger attached to it
        test_trainer = pl.Trainer(accelerator='cpu', logger=False)
        log_dir = exp_manager(
            test_trainer,
            {"create_tensorboard_logger": True, "create_checkpoint_callback": False, "exp_dir": str(tmp_path)},
        )
        assert isinstance(test_trainer.logger, pl.loggers.TensorBoardLogger)

        test_trainer = pl.Trainer(accelerator='cpu', logger=False)
        # Check that a create_wandb_logger=True errors out unless wandb_logger_kwargs is passed.
        with pytest.raises(ValueError):
            log_dir = exp_manager(
                test_trainer,
                {
                    "create_tensorboard_logger": False,
                    "create_checkpoint_callback": False,
                    "exp_dir": str(tmp_path),
                    "create_wandb_logger": True,
                },
            )
        # Check that a WandbLogger is attached to logger if create_wandb_logger=True and wandb_logger_kwargs has name
        # and project
        log_dir = exp_manager(
            test_trainer,
            {
                "create_tensorboard_logger": False,
                "create_checkpoint_callback": False,
                "exp_dir": str(tmp_path),
                "create_wandb_logger": True,
                "wandb_logger_kwargs": {"name": "", "project": "", "offline": True},
            },
        )
        assert isinstance(test_trainer.logger, pl.loggers.WandbLogger)

    @pytest.mark.unit
    def test_checkpoint_configurations(self):
        """ Test that trainer creating modelcheckpoint and asking exp_manager to do it too results in errors, but
        is error free if only one is asked to do so.
        """
        disable_tb_logger = {"create_tensorboard_logger": False}
        test_trainer = pl.Trainer(accelerator='cpu')  # Should create logger and modelcheckpoint
        with pytest.raises(CheckpointMisconfigurationError):  # Fails because both try to create modelcheckpoint
            exp_manager(test_trainer, disable_tb_logger)

        # Should succeed without error
        exp_manager(test_trainer, {"create_checkpoint_callback": False, "create_tensorboard_logger": False})

        test_trainer_2 = pl.Trainer(accelerator='cpu', enable_checkpointing=False)
        exp_manager(test_trainer_2, disable_tb_logger)  # Should succeed without error

    @pytest.mark.unit
    def test_default_log_dir(self):
        """Check the default of ./nemo_experiments/default/datetime works as intended"""
        test_trainer = pl.Trainer(accelerator='cpu', enable_checkpointing=False, logger=False)

        log_dir = exp_manager(test_trainer, {"create_tensorboard_logger": False, "create_checkpoint_callback": False})
        assert (log_dir / "..").resolve() == Path("./nemo_experiments/default/").resolve()
        assert Path("./nemo_experiments").exists()
        assert Path("./nemo_experiments/default/").exists()
        sub_dirs = [x for x in Path("./nemo_experiments/default/").iterdir() if x.is_dir()]
        assert len(sub_dirs) == 1
        assert re.match(r"[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2}", sub_dirs[0].name)

    @pytest.mark.unit
    def test_log_dir_overrides(self, monkeypatch, tmp_path):
        """Check a variety of trainer options with exp_manager"""
        # Checks that explicit_log_dir ignores exp_dir, name, and version
        test_trainer = pl.Trainer(accelerator='cpu', enable_checkpointing=False, logger=False)
        log_dir = exp_manager(test_trainer, {"explicit_log_dir": str(tmp_path / "test_log_dir_overrides")})
        assert log_dir.resolve() == (tmp_path / "test_log_dir_overrides").resolve()
        assert Path(tmp_path).exists()
        assert Path(tmp_path / "test_log_dir_overrides").exists()

        # Checks that exp_manager uses exp_dir, default name, and explicit version
        test_trainer = pl.Trainer(accelerator='cpu', enable_checkpointing=False, logger=False)
        log_dir = exp_manager(test_trainer, {"exp_dir": str(tmp_path / "test_no_name"), "version": 957})
        assert log_dir.resolve() == (tmp_path / "test_no_name" / "default" / "957").resolve()
        assert Path(tmp_path).exists()
        assert Path(tmp_path / "test_no_name" / "default" / "957").exists()

        monkeypatch.delenv(NEMO_ENV_VARNAME_VERSION)
        # Checks that use_datetime_version False toggle works
        test_trainer = pl.Trainer(accelerator='cpu', enable_checkpointing=False, logger=False)
        log_dir = exp_manager(test_trainer, {"exp_dir": str(tmp_path / "test_no_name"), "use_datetime_version": False})
        assert log_dir.resolve() == (tmp_path / "test_no_name" / "default" / "version_0").resolve()
        assert Path(tmp_path).exists()
        assert Path(tmp_path / "test_no_name" / "default" / "version_0").exists()

        monkeypatch.delenv(NEMO_ENV_VARNAME_VERSION)
        # Checks that use_datetime_version False toggle works and version increments
        test_trainer = pl.Trainer(accelerator='cpu', enable_checkpointing=False, logger=False)
        log_dir = exp_manager(test_trainer, {"exp_dir": str(tmp_path / "test_no_name"), "use_datetime_version": False})
        assert log_dir.resolve() == (tmp_path / "test_no_name" / "default" / "version_1").resolve()
        assert Path(tmp_path).exists()
        assert Path(tmp_path / "test_no_name" / "default" / "version_1").exists()

    @pytest.mark.unit
    def test_resume(self, tmp_path):
        """ Tests the resume capabilities of exp_manager"""
        test_trainer = pl.Trainer(accelerator='cpu', enable_checkpointing=False, logger=False)

        # Error because explicit_log_dir does not exist
        with pytest.raises(NotFoundError):
            exp_manager(
                test_trainer,
                {
                    "exp_dir": str(tmp_path / "test_resume"),
                    "resume_if_exists": True,
                    "explicit_log_dir": "Does_not_exist",
                },
            )

        # Error because checkpoints folder does not exist
        with pytest.raises(NotFoundError):
            exp_manager(test_trainer, {"resume_if_exists": True, "exp_dir": str(tmp_path / "test_resume")})

        # No error because we tell exp_manager to ignore notfounderror
        exp_manager(
            test_trainer,
            {
                "resume_if_exists": True,
                "exp_dir": str(tmp_path / "test_resume_2"),
                "resume_ignore_no_checkpoint": True,
            },
        )

        test_trainer = pl.Trainer(accelerator='cpu', enable_checkpointing=False, logger=False)
        Path(tmp_path / "test_resume" / "default" / "version_0" / "checkpoints").mkdir(parents=True)
        # Error because checkpoints do not exist in folder
        with pytest.raises(NotFoundError):
            exp_manager(
                test_trainer,
                {
                    "resume_if_exists": True,
                    "explicit_log_dir": str(tmp_path / "test_resume" / "default" / "version_0"),
                },
            )

        Path(tmp_path / "test_resume" / "default" / "version_0" / "checkpoints" / "mymodel--end.ckpt").touch()
        # Error because *end.ckpt is in folder indicating that training has already finished
        with pytest.raises(ValueError):
            exp_manager(
                test_trainer,
                {
                    "resume_if_exists": True,
                    "explicit_log_dir": str(tmp_path / "test_resume" / "default" / "version_0"),
                },
            )

        Path(tmp_path / "test_resume" / "default" / "version_0" / "checkpoints" / "mymodel--end.ckpt").unlink()
        Path(tmp_path / "test_resume" / "default" / "version_0" / "checkpoints" / "mymodel--last.ckpt").touch()
        Path(tmp_path / "test_resume" / "default" / "version_0" / "checkpoints" / "mymodel2--last.ckpt").touch()
        # Error because multiple *last.ckpt is in folder. If more than one, don't know which to restore
        with pytest.raises(ValueError):
            exp_manager(
                test_trainer,
                {
                    "resume_if_exists": True,
                    "explicit_log_dir": str(tmp_path / "test_resume" / "default" / "version_0"),
                },
            )

        # Finally succeed
        Path(tmp_path / "test_resume" / "default" / "version_0" / "checkpoints" / "mymodel2--last.ckpt").unlink()
        log_dir = exp_manager(
            test_trainer,
            {"resume_if_exists": True, "explicit_log_dir": str(tmp_path / "test_resume" / "default" / "version_0")},
        )
        checkpoint = Path(tmp_path / "test_resume" / "default" / "version_0" / "checkpoints" / "mymodel--last.ckpt")
        assert (
            Path(test_trainer._checkpoint_connector.resume_from_checkpoint_fit_path).resolve() == checkpoint.resolve()
        )

        # Succeed again and make sure that run_0 exists and previous log files were moved
        test_trainer = pl.Trainer(accelerator='cpu', enable_checkpointing=False, logger=False)
        exp_manager(test_trainer, {"resume_if_exists": True, "explicit_log_dir": str(log_dir)})
        checkpoint = Path(tmp_path / "test_resume" / "default" / "version_0" / "checkpoints" / "mymodel--last.ckpt")
        assert (
            Path(test_trainer._checkpoint_connector.resume_from_checkpoint_fit_path).resolve() == checkpoint.resolve()
        )
        prev_run_dir = Path(tmp_path / "test_resume" / "default" / "version_0" / "run_0")
        assert prev_run_dir.exists()
        prev_log = Path(tmp_path / "test_resume" / "default" / "version_0" / "run_0" / "lightning_logs.txt")
        assert prev_log.exists()

    @pytest.mark.unit
    def test_nemo_checkpoint_save_best_model_1(self, tmp_path):
        test_trainer = pl.Trainer(accelerator='cpu', enable_checkpointing=False, logger=False, max_epochs=4)
        exp_manager(
            test_trainer,
            {"checkpoint_callback_params": {"save_best_model": True}, "explicit_log_dir": str(tmp_path / "test")},
        )
        model = ExampleModel()
        test_trainer.fit(model)

        assert Path(str(tmp_path / "test" / "checkpoints" / "default.nemo")).exists()

        model = ExampleModel.restore_from(str(tmp_path / "test" / "checkpoints" / "default.nemo"))
        assert float(model(torch.tensor([1.0, 1.0], device=model.device))) == 0.0

    @pytest.mark.unit
    def test_nemo_checkpoint_save_best_model_2(self, tmp_path):
        test_trainer = pl.Trainer(accelerator='cpu', enable_checkpointing=False, logger=False, max_epochs=4)
        exp_manager(
            test_trainer, {"explicit_log_dir": str(tmp_path / "test")},
        )
        model = ExampleModel()
        test_trainer.fit(model)

        assert Path(str(tmp_path / "test" / "checkpoints" / "default.nemo")).exists()

        model = ExampleModel.restore_from(str(tmp_path / "test" / "checkpoints" / "default.nemo"))
        assert math.fabs(float(model(torch.tensor([1.0, 1.0], device=model.device))) - 0.03) < 1e-5

    @pytest.mark.unit
    def test_nemo_checkpoint_always_save_nemo(self, tmp_path):
        test_trainer = pl.Trainer(accelerator='cpu', enable_checkpointing=False, logger=False, max_epochs=4)
        exp_manager(
            test_trainer,
            {
                "checkpoint_callback_params": {"save_best_model": True, "always_save_nemo": True},
                "explicit_log_dir": str(tmp_path / "test"),
            },
        )
        model = ExampleModel()
        test_trainer.fit(model)

        assert Path(str(tmp_path / "test" / "checkpoints" / "default.nemo")).exists()

        model = ExampleModel.restore_from(str(tmp_path / "test" / "checkpoints" / "default.nemo"))
        assert float(model(torch.tensor([1.0, 1.0], device=model.device))) == 0.0

    @pytest.mark.unit
    def test_nemo_checkpoint_make_checkpoint_dir(self, tmp_path):
        test_trainer = pl.Trainer(
            accelerator='cpu', enable_checkpointing=False, logger=False, max_epochs=4, check_val_every_n_epoch=5
        )
        exp_manager(
            test_trainer,
            {
                "checkpoint_callback_params": {"save_best_model": True, "always_save_nemo": True},
                "explicit_log_dir": str(tmp_path / "test"),
            },
        )
        model = ExampleModel()
        test_trainer.fit(model)

        assert Path(str(tmp_path / "test" / "checkpoints" / "default.nemo")).exists()

    @pytest.mark.unit
    def test_nemo_checkpoint_restore_model(self, tmp_path):
        test_trainer = pl.Trainer(accelerator='cpu', enable_checkpointing=False, logger=False, max_epochs=4)
        exp_manager(
            test_trainer,
            {
                "checkpoint_callback_params": {"save_top_k": 1, "save_last": True},
                "explicit_log_dir": str(tmp_path / "test"),
            },
        )
        model = ExampleModel()
        test_trainer.fit(model)

        checkpoint = list(Path(str(tmp_path / "test" / "checkpoints")).glob("*.ckpt"))
        # Make sure that only the best and last checkpoint is saved
        assert len(checkpoint) == 2
        assert math.fabs(float(model(torch.tensor([1.0, 1.0], device=model.device))) - 0.03) < 1e-5

        test_trainer = pl.Trainer(accelerator='cpu', enable_checkpointing=False, logger=False, max_epochs=5)
        exp_manager(
            test_trainer,
            {
                "checkpoint_callback_params": {"save_top_k": 1, "save_last": False},
                "explicit_log_dir": str(tmp_path / "test"),
                "resume_if_exists": True,
                "resume_past_end": True,
            },
        )
        model = DoNothingModel()
        model.l1.weight = torch.nn.Parameter(torch.tensor((0.0, 0.0)).unsqueeze(0))
        model.l1.bias = torch.nn.Parameter(torch.tensor(1.0))
        assert math.fabs(float(model(torch.tensor([1.0, 1.0], device=model.device))) - 1.0) < 1e-5

        test_trainer.fit(model)
        assert math.fabs(float(model(torch.tensor([1.0, 1.0], device=model.device))) - 0.03) < 1e-5
