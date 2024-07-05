import pytest
import pytorch_lightning as pl
from torch import nn

from nemo.lightning.pytorch.callbacks.model_transform import ModelTransform


class TestModelTransformCallback:
    @pytest.fixture
    def callback(self):
        return ModelTransform()

    @pytest.fixture
    def pl_module(self):
        return MockLightningModule()

    @pytest.fixture
    def trainer(self):
        return pl.Trainer()

    def test_setup_stores_transform(self, callback, pl_module, trainer, caplog):
        callback.setup(trainer, pl_module, 'fit')

        assert callback.model_transform is not None, "callback.model_transform should be set after setup"
        assert hasattr(
            callback.model_transform, '__num_calls__'
        ), "callback.model_transform should have __num_calls__ attribute"
        assert callback.model_transform.__num_calls__ == 0, "callback.model_transform should not have been called yet"
        assert pl_module.model_transform == callback.model_transform, "pl_module.model_transform should be updated"


class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)


class MockLightningModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = MockModel()
        self.model_transform = lambda m: nn.Sequential(m, nn.ReLU())

    def forward(self, x):
        return self.model(x)
