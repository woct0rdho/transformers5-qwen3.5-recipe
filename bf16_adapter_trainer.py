from transformers import Trainer

from gdn_stock_checkpoints import loading_optimizer_state, saving_optimizer_state


class BF16AdapterTrainer(Trainer):
    """Keep PEFT adapters in BF16 when Trainer restores a checkpoint.

    It also keeps the checkpoint in transformers' value-head layout while the model runs in
    llama.cpp's tiled order (`gdn_stock_checkpoints`): `optimizer.pt` is written and read in
    the layout its `adapter_model.safetensors` declares, so resume and stock tools agree.
    """

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        checkpoint_model = model if model is not None else self.model
        load_adapter = getattr(checkpoint_model, "load_adapter", None)
        if load_adapter is None:
            return super()._load_from_checkpoint(resume_from_checkpoint, model=model)

        def load_adapter_without_upcast(*args, **kwargs):
            kwargs["autocast_adapter_dtype"] = False
            return load_adapter(*args, **kwargs)

        checkpoint_model.__dict__["load_adapter"] = load_adapter_without_upcast
        return super()._load_from_checkpoint(
            resume_from_checkpoint, model=checkpoint_model
        )

    def _save_optimizer_and_scheduler(self, output_dir):
        with saving_optimizer_state(self.optimizer, self.model):
            return super()._save_optimizer_and_scheduler(output_dir)

    def _load_optimizer_and_scheduler(self, checkpoint):
        with loading_optimizer_state(self.optimizer, self.model):
            return super()._load_optimizer_and_scheduler(checkpoint)
