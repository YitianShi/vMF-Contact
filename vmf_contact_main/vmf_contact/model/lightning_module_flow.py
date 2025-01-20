from typing import Any, Dict, Tuple

import pytorch_lightning as pl
import torch
from torch import optim

from ..nn import BayesianLoss, vmfContact, UncertaintyEstimator

Batch = Tuple[torch.Tensor, torch.Tensor]


class vmfContactFlowLightningModule(pl.LightningModule):
    def __init__(
        self,
        args,
        # loss parameters
        flow_loss_coeff=1e-2,
    ) -> None:

        super().__init__()
        self.save_hyperparameters()

        self.args = args
        self.prob_baseline = args.prob_baseline

        self.flow_loss_coeff = flow_loss_coeff

        self.debug = args.debug
        self.batch_size = args.batch_size
        self.automatic_optimization = False

        self.learning_rate_flow = 3e-4
        self.gradient_accumulation_steps = args.gradient_accumulation_steps

        self.model = vmfContact(
            args=args,
            image_size=args.image_size,
            embedding_dim=args.embedding_dim,
            scale=args.scale,
            prob_baseline=args.prob_baseline,
            pcd_with_rgb=args.pcd_with_rgb,
        )

        self.uncertainty_estimator = UncertaintyEstimator(
            hidden_features=args.hidden_feat_flow,
            embedding_dim=args.embedding_dim,
            flow_layers=args.flow_layers,
            certainty_budget=args.certainty_budget,
            prob_baseline=args.prob_baseline,
            flow_type=args.flow_type
        )

        self.loss = BayesianLoss(args.entropy_weight)

    def training_step(self, batch, _batch_idx: int):

        self.forward_and_loss(batch)

        self.log_dict(self.losses, sync_dist=True, batch_size=self.batch_size, prog_bar=True)
        self.log(
            "train/lr",
            self.lr_schedulers().get_last_lr()[0],
            sync_dist=True,
            prog_bar=True,
        )
        loss = sum(list(self.losses.values()))

        if (_batch_idx+1) % self.gradient_accumulation_steps == 0:
            opt = self.optimizers()
            self.manual_backward(loss)
            # for name, parameter in self.uncertainty_estimator.flow.named_parameters():
            #     if parameter.grad is not None:
            #         print(name)
            opt.step()
            opt.zero_grad()

    def validation_step(self, batch, _batch_idx):
        self.model.eval()
        self.forward_and_loss(batch)
        loss = sum(list(self.losses.values()))

        self.log_dict(self.losses, sync_dist=True, batch_size=self.batch_size)
        self.log(
            "val/loss", loss, sync_dist=True, prog_bar=True, batch_size=self.batch_size
        )

        return loss

    def test_step(self, batch, _batch_idx: int):
        pred, _ = self.forward_and_loss(batch, test=True)
        return pred

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer_grouped_parameters = [
            {
                "params": [],
                "weight_decay": 0,
                "lr": self.learning_rate_flow,
                "betas": (0.9, 0.999),
                "eps": 1e-08,
            },
        ]

        for _, param in self.uncertainty_estimator.named_parameters():
            optimizer_grouped_parameters[0]["params"].append(param)

        optimizer = optim.Adam(optimizer_grouped_parameters, lr=self.learning_rate_flow)

        config: Dict[str, Any] = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=0.5,
                    patience=50,
                    threshold=1e-3,
                    min_lr=1e-7,
                ),
                "monitor": "train/loss",
            },
        }

        return config

    def forward_and_loss(self, batch) -> Dict[str, torch.Tensor]:

        self.model.eval()

        input_batch, gt_batch = batch
        pcd = input_batch["pcd"]
        self.batch_size = pcd.shape[0]

        #if self.prob_baseline == "post":
        self.uncertainty_estimator.update_lipschitz()

        # List of ground-truth contact points positions
        self.gt_pos_bt = list(gt_sample["gt_pt"] for gt_sample in gt_batch)
        with torch.no_grad():
            pred = self.model(pcd, self.gt_pos_bt, debug=self.debug)

        self.losses = {}
        # if self.prob_baseline == "post":
        self._flow_loss(pred)

        return pred


    def _flow_loss(self, pred, groups=20):
        self.uncertainty_estimator.train()
        # torch.autograd.set_detect_anomaly(True)
        features = pred["cp_features"]

        # Compute loss
        features = features.view(-1, features.shape[-1])  # .to(self.model.flow_device)
        loss = self.uncertainty_estimator.forward_kld(features)

        # Make layers Lipschitz continuous
        self.losses["flow_loss"] = loss * self.flow_loss_coeff
