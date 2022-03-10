# Speech Classification

This directory contains example scripts to train speech classification and voice activity detection models. 

# Model execution overview

The training scripts in this directory execute in the following order. When preparing your own training-from-scratch / fine-tuning scripts, please follow this order for correct training/inference.

```mermaid

graph TD
    A[Hydra Overrides + Yaml Config] --> B{Config}
    B --> |Init| C[Trainer]
    C --> D[ExpManager]
    B --> D[ExpManager]
    C --> E[Model]
    B --> |Init| E[Model]
    E --> |Constructor| F(Change Labels)
    F --> G(Setup Train + Validation + Test Data loaders)
    G --> H(Setup Optimization)
    H --> I[Maybe init from pretrained]
    I --> J["trainer.fit(model)"]
```

During restoration of the model, you may pass the Trainer to the restore_from / from_pretrained call, or set it after the model has been initialized by using `model.set_trainer(Trainer)`.