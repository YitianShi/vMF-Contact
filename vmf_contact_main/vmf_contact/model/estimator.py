from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from ..datasets import DataModule
from ..nn import vmfContact

from .lightning_module import vmfContactLightningModule
from .lightning_module_flow import vmfContactFlowLightningModule

import os
#get current file directory
current_file_folder = os.path.dirname(os.path.abspath(__file__))

logger = logging.getLogger(__name__)


class vmfContactModule():
    """
    Estimator for the vmfContact.
    """

    #: The input size of the model.
    model_: vmfContactLightningModule
    input_size_: torch.Size

    def __init__(
        self,
        args,
        *,
        finetune: bool = True,
        user_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            encoder: The type of encoder to use which maps the input to the latent space.
            flow: The type of flow which produces log-probabilities from the latent
                representations.
            certainty_budget: The certainty budget to use to scale the log-probabilities produced
                by the normalizing flow.
            learning_rate: The learning rate to use for training encoder, flow, and linear output
                layer. Applies to warm-up, actual training, and fine-tuning.
            learning_rate_decay: Whether to use a learning rate decay by reducing the learning rate
                when the validation loss plateaus.
            flow_finetune: The number of epochs to run warm-up for. Should be used if the latent
                space is high-dimensional and/or the normalizing flow is complex, i.e. consists of
                many layers.
            finetune: Whether to run fine-tuning after the main training loop. May be set to
                ``False`` to speed up the overall training time if the data is simple. Otherwise,
                it should be kept as ``True`` to improve out-of-distribution detection.
            ensemble_size: The number of NatPN models to ensemble for the final predictions. This
                constructs a Natural Posterior Ensemble which trains multiple NatPN models
                independently and combines their predictions via Bayesian combination. By default,
                this is set to ``None`` which does not create a NatPE.
            user_params: Additional parameters which are passed to the PyTorch Ligthning
                trainer. These parameters apply to all fitting runs as well as testing.
        """
        
        overwrite_params=dict(
                enable_checkpointing=True,
                enable_progress_bar=True,
            )

        self.trainer_params_user = user_params
        self.trainer_params = {
            **dict(
                logger=False,
                log_every_n_steps=1,
                enable_progress_bar=logger.getEffectiveLevel() <= logging.INFO,
                enable_checkpointing=logger.getEffectiveLevel() <= logging.DEBUG,
                enable_model_summary=logger.getEffectiveLevel() <= logging.DEBUG),
            **user_params,
            **overwrite_params,
        }

        self.args = args
        self.flow_finetune = args.flow_finetune
        self.finetune = args.run_finetuning

    def trainer(self, **kwargs: Any) -> pl.Trainer:
        """
        Returns the trainer as configured by the estimator. Typically, this method is only called
        by functions in the estimator.

        Args:
            kwargs: Additional arguments that override the trainer arguments registered in the
                initializer of the estimator.

        Returns:
            A fully initialized PyTorch Lightning trainer.

        Note:
            This function should be preferred over initializing the trainer directly. It ensures
            that the returned trainer correctly deals with LightKit components that may be
            introduced in the future.
        """
        return pl.Trainer(**{**self.trainer_params, **kwargs})

    # ---------------------------------------------------------------------------------------------
    # RUNNING THE MODEL

    def fit(self, data: DataModule) -> vmfContactModule:
        """
        Fits the network with the provided data. Fitting sequentially runs
        warm-up (if ``self.flow_finetune > 0``), the main training loop, and fine-tuning (if
        ``self.finetune == True``).

        Args:
            data: The data to fit the model with.

        Returns:
            The estimator whose ``model_`` property is set.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            if self.args.eval:
                self._val_model(data, Path(tmp_dir), ckpt=self.args.ckpt)
            else:
                self._fit_model(data, Path(tmp_dir))

        # Assign additional fitted attributes
        self.input_size_ = data.input_size
        self.output_type_ = data.output_type

        # Return self
        return self

    def score(self, data: DataModule) -> Dict[str, float]:
        """
        Measures the model performance on the given data.

        Args:
            data: The data for which to measure the model performance.

        Returns:
            A dictionary mapping metrics to their values. This dictionary includes a measure of
            accuracy (`"accuracy"` for classification and `"rmse"` for regression) and a
            calibration measure (`"brier_score"` for classification and `"calibration"` for
            regression).
        """
        logger.info("Evaluating on test set...")
        module = vmfContactLightningModule.load_from_checkpoint(
            self.model_.best_model_path, strict=False,
            debug=self.args.debug
        )
        out = self.trainer().test(module, data, verbose=False)
        return {k.split("/")[1]: v for k, v in out[0].items()}

    # ---------------------------------------------------------------------------------------------
    # PERSISTENCE

    @property
    def persistent_attributes(self) -> List[str]:
        return [k for k in self.__annotations__ if k != "model_"]

    def save_parameters(self, path: Path) -> None:
        params = {
            k: (
                v
                if k != "trainer_params"
                else {
                    kk: vv
                    for kk, vv in cast(Dict[str, Any], v).items()
                    if kk != "logger"
                }
            )
            for k, v in self.get_params().items()
        }
        data = json.dumps(params, indent=4)
        with (path / "params.json").open("w+") as f:
            f.write(data)

    def save_attributes(self, path: Path) -> None:
        super().save_attributes(path)
        torch.save(self.model_.state_dict(), path / "parameters.pt")

    def load_attributes(self, path: Path) -> None:
        super().load_attributes(path)
        parameters = torch.load(path / "parameters.pt")
        model = vmfContact(
            args=self.args,
            backbone=self.args.backbone,
            image_size=self.args.image_size,
            embedding_dim=self.args.embedding_dim,
            scale=self.args.scale,
            pcd_with_rgb=self.args.pcd_with_rgb,
        )
        model.load_state_dict(parameters)
        self.model_ = model

    # ---------------------------------------------------------------------------------------------
    # UTILS

    def _fit_model(self, data: DataModule, tmp_dir: Path) -> vmfContact:
        level = logging.getLogger("pytorch_lightning").getEffectiveLevel()

        # Run training
        trainer_checkpoint = ModelCheckpoint(
            f"logs/training/{self.args.point_backbone}", monitor="val/loss", mode="min"
        )

        logging.getLogger("pytorch_lightning").setLevel(
            logging.INFO if self.flow_finetune == 0 else level
        )
        trainer = self.trainer(
            # accumulate_grad_batches=data.gradient_accumulation_steps,
            callbacks=[trainer_checkpoint],
            enable_model_summary=True,
        )
        logging.getLogger("pytorch_lightning").setLevel(level)

        logger.info("Running training...")

        main_module, ckpt_loaded = self.module_loader(self.args.ckpt)

        # Train main module
        if not ckpt_loaded or self.flow_finetune == 0:
            logger.warning("Train main module from scratch.")
            trainer.fit(main_module, data)
            best_module = vmfContactLightningModule.load_from_checkpoint(
                trainer_checkpoint.best_model_path, strict=False,
                debug=self.args.debug
            )
        else:
            best_module = main_module

        # Fine-tune flow module
        if self.flow_finetune > 0:
            logger.info("Running finetuning for flow module ...")
            flow_module = vmfContactFlowLightningModule(args=self.args)
            flow_module.model.load_state_dict(best_module.model.state_dict())
            trainer.fit(flow_module, data)
            # Load the uncertainty estimator from the flow module back into the main module
            best_module.uncertainty_estimator.load_state_dict(flow_module.uncertainty_estimator.state_dict())
            del flow_module
    
        # Run fine-tuning
        if self.finetune:
            finetune_checkpoint = ModelCheckpoint(
                f"{current_file_folder}/../../../logs/training/{self.args.point_backbone}", monitor="val/loss", mode="min"
            )
            trainer = self.trainer(
                accumulate_grad_batches=data.gradient_accumulation_steps,
                callbacks=[finetune_checkpoint],
            )

            logger.info("Running fine-tuning...")
            finetune_module = best_module
            trainer.fit(finetune_module, data)

            # Return model
            return vmfContactLightningModule.load_from_checkpoint(
                finetune_checkpoint.best_model_path, strict=False,
                debug=self.args.debug
            ).model
        return cast(vmfContact, best_module.model)
    

    def _val_model(self, data: DataModule, tmp_dir: Path, ckpt) -> vmfContact:
        level = logging.getLogger("pytorch_lightning").getEffectiveLevel()

        # Run training
        trainer_checkpoint = ModelCheckpoint(
            "logs/val", monitor="val/loss", mode="min"
        )

        logging.getLogger("pytorch_lightning").setLevel(
            logging.INFO if self.flow_finetune == 0 else level
        )
        trainer = self.trainer(
            # accumulate_grad_batches=data.gradient_accumulation_steps,
            callbacks=[trainer_checkpoint],
            enable_model_summary=self.flow_finetune == 0,
        )
        logging.getLogger("pytorch_lightning").setLevel(level)

        logger.info("Running validation...")

        main_module, ckpt_loaded = self.module_loader(ckpt=ckpt)

        if ckpt_loaded:
            data.setup("validate")
            trainer.validate(main_module, data.val_dataloader())
        else:
            raise ValueError("No checkpoint found to resume from, please check.")
    

    def module_loader(self, ckpt = None):
        if ckpt is not None:
            ckpt = Path(f"{current_file_folder}/../../../logs/training/{self.args.point_backbone}/{ckpt}.ckpt")
            logger.info(f"Resuming from checkpoint: {ckpt}")
            main_module = vmfContactLightningModule.load_from_checkpoint(ckpt, strict=False, debug=self.args.debug, args=self.args)
            ckpt_loaded = True
        else:
            logger.warning(
                "No checkpoint found to resume from, train contact 3d from scratch."
            )
            main_module = vmfContactLightningModule(args=self.args)
            ckpt_loaded = False
        return main_module, ckpt_loaded
            
