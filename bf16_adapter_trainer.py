from transformers import Trainer


class BF16AdapterTrainer(Trainer):
    """Keep PEFT adapters in BF16 when Trainer restores a checkpoint."""

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        checkpoint_model = model if model is not None else self.model
        load_adapter = getattr(checkpoint_model, "load_adapter", None)
        if load_adapter is None:
            return super()._load_from_checkpoint(resume_from_checkpoint, model=model)

        def load_adapter_without_upcast(*args, **kwargs):
            kwargs["autocast_adapter_dtype"] = False
            return load_adapter(*args, **kwargs)

        checkpoint_model.load_adapter = load_adapter_without_upcast
        return super()._load_from_checkpoint(
            resume_from_checkpoint, model=checkpoint_model
        )
