"""Training — Lightning modules per trainable component.

Only the OCR component is trained in A2. TrOCR is an encoder-decoder, so the loss is
ordinary sequence cross-entropy over the label ids. Two details are load-bearing and
neither is visible from the loss curve:

* padding must be masked to -100, or the model is rewarded for predicting pad and learns
  to stop early;
* ``from_pretrained`` hands back a model in eval mode and Lightning does not put it back
  into train mode, so without ``on_train_start`` every dropout layer is inert;
* the model's own ``.loss`` shifts the labels a second time -- see ``_loss``.

The learning rate was measured, not inherited: over a 160-step probe (mean loss of the
last 40 steps) 5e-6 gave 5.76, 5e-5 gave 2.65 and 2e-4 gave 4.47. 5e-6 is the rate a 334M
model wants; on the 62M model actually used here it undertrains badly enough that the
recogniser drops most of the words in a line.
"""

from __future__ import annotations

import lightning as pl
import torch
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig

from ..logging_conf import get_logger

log = get_logger(__name__)


class LitComponent(pl.LightningModule):
    """Wrap enhancer / OCR / retriever training. Here: the TrOCR fine-tune."""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        tc = cfg.get("ocr_train", {})
        name = tc.get("base_model", "microsoft/trocr-base-printed")
        self.lr = float(tc.get("lr", 5e-6))
        self.weight_decay = float(tc.get("weight_decay", 0.01))
        self.warmup = int(tc.get("warmup_steps", 200))
        self.max_len = int(tc.get("max_target_length", 64))

        self.processor = TrOCRProcessor.from_pretrained(name)
        self.model = VisionEncoderDecoderModel.from_pretrained(name)
        # ProcessorMixin builds its sub-processors dynamically, so this is invisible to mypy.
        self.tokenizer = self.processor.tokenizer  # type: ignore[attr-defined]

        # The pretrained checkpoint ships without these set, and generate() then produces
        # empty strings rather than failing.
        self.model.config.decoder_start_token_id = self.tokenizer.cls_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.eos_token_id = self.tokenizer.sep_token_id
        self.model.config.vocab_size = self.model.config.decoder.vocab_size

        if bool(tc.get("freeze_encoder", False)):
            for p in self.model.encoder.parameters():
                p.requires_grad = False

    def on_train_start(self) -> None:
        # ``from_pretrained`` returns the model in eval mode and Lightning does not undo that
        # (it warns: "Found N module(s) in eval mode"). Left alone, every dropout layer in the
        # 334M-parameter model is inert for the whole run.
        self.model.train()

    def _prepare(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        pixels = self.processor(images=batch["image"], return_tensors="pt").pixel_values
        tok = self.tokenizer(
            batch["text"],
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        labels = tok.input_ids.masked_fill(tok.attention_mask == 0, -100)
        return pixels.to(self.device), labels.to(self.device)

    def _loss(self, pixels: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Cross-entropy at matching positions -- deliberately NOT the model's own ``.loss``.

        Passing ``labels`` makes VisionEncoderDecoderModel build ``decoder_input_ids`` by
        shifting them right, which is what we want, but it then scores them with
        ``ForCausalLMLoss``, which shifts *again* ("loss_type=None ... using the default
        loss" in the logs). The two shifts compose: the model is trained to predict the
        token after next, and generating autoregressively it then skips every other token.
        The symptom is a recogniser that reads "and cold soft water, immerse the linen and
        boil" as "and soft, the and" -- every second word, in order, at CER 0.70.
        """
        logits = self.model(pixel_values=pixels, labels=labels).logits
        return torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
        )

    def training_step(self, batch: dict, idx: int) -> torch.Tensor:
        loss = self._loss(*self._prepare(batch))
        self.log("train_loss", loss, prog_bar=True, batch_size=len(batch["text"]))
        return loss

    def validation_step(self, batch: dict, idx: int) -> torch.Tensor:
        loss = self._loss(*self._prepare(batch))
        self.log("val_loss", loss, prog_bar=True, batch_size=len(batch["text"]))
        return loss

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        opt = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        # Linear warmup then constant. The run is a few thousand steps; a full cosine decay
        # would spend most of them at a rate too low to move a 334M-parameter model.
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda step: min(1.0, (step + 1) / max(1, self.warmup))
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

    def save(self, out_dir: str) -> None:
        self.model.save_pretrained(out_dir)
        self.processor.save_pretrained(out_dir)
        log.info("saved fine-tuned OCR model to %s", out_dir)
