"""Trainer for this repository's BF16 LoRA adapters.

`BF16AdapterTrainer` keeps PEFT adapters in BF16 when Trainer restores a checkpoint, and keeps
the checkpoint in transformers' value-head layout while the model runs in llama.cpp's tiled
order (`gdn_stock_checkpoints`): `optimizer.pt` is written and read in the layout its
`adapter_model.safetensors` declares, so resume and stock tools agree.

It also installs `AdamW8bitSR` (`adamw_8bit_sr`) instead of the stock optimizer. That one keeps
bitsandbytes' 8-bit blockwise state format byte for byte, and does the AdamW update, the
decoupled weight decay and both stochastic roundings in a single Triton kernel: without the
stochastic rounding a `lr*wd = 1e-6` decay, and every Adam step below half a BF16 ulp, is
discarded by the parameter store. `moe_weight_decay` gives the packed routed experts a smaller
decay than the dense adapters, for the reason described in that module.
"""

from transformers import Trainer
from transformers.utils import logging

from adamw_8bit_sr import AdamW8bitSR
from gdn_stock_checkpoints import loading_optimizer_state, saving_optimizer_state

logger = logging.get_logger(__name__)


class BF16AdapterTrainer(Trainer):
    """Keep PEFT adapters in BF16 across resume, and step them with `AdamW8bitSR`."""

    def __init__(self, *args, moe_weight_decay: float | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.moe_weight_decay = moe_weight_decay

    def create_optimizer(self, model=None):
        if self.optimizer is not None:
            return self.optimizer

        opt_model = self.model if model is None else model
        assert opt_model is not None
        self.optimizer = AdamW8bitSR(
            opt_model.named_parameters(),
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
            weight_decay=self.args.weight_decay,
            routed_expert_decay=self.moe_weight_decay,
        )
        logger.info(
            "optimizer: AdamW8bitSR, lr %g, dense weight_decay %g, routed-expert weight_decay %s, "
            "8-bit state for %d parameter(s)",
            self.args.learning_rate,
            self.args.weight_decay,
            "same as dense"
            if self.moe_weight_decay is None
            else f"{self.moe_weight_decay:g}",
            len(self.optimizer.param_groups[0]["params"]),
        )
        return self.optimizer

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
        if self.optimizer is None:
            self.create_optimizer()
        with loading_optimizer_state(self.optimizer, self.model):
            return super()._load_optimizer_and_scheduler(checkpoint)
