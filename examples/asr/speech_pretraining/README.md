# Speech Pre-training via Self Supervised Learning

This directory contains example scripts to train ASR models using various Self Supervised Losses. 

The model's pretrained here can further be finetuned on specific labeled data in further steps.

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
    E --> |Constructor| G(Setup Train + Validation Data loaders)
    G --> H(Setup Optimization)
    H --> I[Maybe init from pretrained]
    I --> J["trainer.fit(model)"]

```

During restoration of the model, you may pass the Trainer to the restore_from / from_pretrained call, or set it after the model has been initialized by using `model.set_trainer(Trainer)`.