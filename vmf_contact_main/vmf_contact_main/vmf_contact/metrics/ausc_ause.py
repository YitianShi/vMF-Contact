from typing import Any, cast, List
import torch
import torchmetrics
import torchmetrics.functional as M

uncert = {"aleatoric", "epistemic", "total"}

class AUSC(torchmetrics.Metric):
    """
    Computes the area under the precision recall curve.
    """

    def __init__(self, keys: List[str], dist_sync_fn: Any = None, baseline_unc = None, intervals = 20) -> None:
        super().__init__(dist_sync_fn=dist_sync_fn)
        self.baseline_unc = baseline_unc
        self.intervals = intervals
        self.keys = keys
        for key in keys:
            setattr(self, f"{key}_pred", [])
            setattr(self, f"{key}_gt", [])
            self.add_state(f"{key}_pred", [], dist_reduce_fx="cat")
            self.add_state(f"{key}_gt", [], dist_reduce_fx="cat")
        if baseline_unc is not None:
            self.add_state("kappa_lh", [], dist_reduce_fx="cat")
            if baseline_unc == "post":
                self.add_state("kappa_post", [], dist_reduce_fx="cat")
        self.results = {}

    def update(self, values: torch.Tensor, targets: torch.Tensor = None, key: str = None) -> None:
        if key == "kappa_lh":
            self.kappa_lh.append(values)
        elif key == "kappa_post":
            self.kappa_post.append(values)
        else:
            getattr(self, f"{key}_pred").append(values)
            getattr(self, f"{key}_gt").append(targets)

    def compute(self) -> torch.Tensor:
        for key in self.keys:
            

            if key == "baseline" and self.baseline_unc is not None:

                baseline_gt = getattr(self, f"{key}_gt")
                baseline = getattr(self, f"baseline_pred")
                
                if self.baseline_unc == "post" or self.baseline_unc == "lh":
                    if self.baseline_unc == "lh":
                        unc_key_list = ["aleatoric"]
                        kappa_list = [self.kappa_lh]
                    elif self.baseline_unc == "post":
                        unc_key_list = ["aleatoric", "epistemic", "total"]
                        kappa_list = [self.kappa_lh, self.kappa_post-self.kappa_lh, self.kappa_post]
                    
                    quants = torch.tensor([1./self.intervals*t for t in range(0,self.intervals)], device=baseline.device) 

                    for unc_key, kappa_type in zip(unc_key_list, kappa_list):
                        kappa = kappa_type.squeeze(-1)

                        # Angular error
                        true_angular_error = torch.acos(
                                torch.clamp(
                                    torch.sum(baseline * baseline_gt, dim=-1), min=-1.0, max=1.0
                                ), 
                            ) * 180.0 / torch.pi
                        
                        # Uncertaitny AUSC
                        thresholds = [torch.quantile(kappa, q) for q in quants]
                        subsets = [kappa >= threshold for threshold in thresholds]
                        sparse_curve = torch.stack([true_angular_error[subset].mean() for subset in subsets])
                        
                        # True AUSC
                        thresholds_true_e = [torch.quantile(-true_angular_error, q) for q in quants]
                        subsets_true_e = [-true_angular_error >= threshold for threshold in thresholds_true_e]
                        sparse_curve_true = torch.stack([true_angular_error[subset].mean() for subset in subsets_true_e])

                        # AUSC and AUSE
                        self.results[f"baseline_ausc_{unc_key}"] = torch.trapz(sparse_curve, x=quants)
                        self.results[f"baseline_ause_{unc_key}"] = self.results[f"baseline_ausc_{unc_key}"] - torch.trapz(sparse_curve_true, x=quants)
                else:
                    self.results["baseline_error"] = M.cosine_similarity(getattr(self, f"{key}_pred"), baseline_gt, reduction="mean")

            elif key == "graspness":
                graspness_threshold = [.5, .75, .9]
                graspness = getattr(self, f"{key}_pred").sigmoid()
                graspness_gt = getattr(self, f"{key}_gt")
                for threshold in graspness_threshold:
                    graspness_gt_thresholded = (graspness_gt > threshold).long()
                    self.results[f"graspness_auroc_{threshold}"] = M.auroc(graspness, graspness_gt_thresholded, task="binary")
                    self.results[f"graspness_ap_{threshold}"] = M.average_precision(graspness, graspness_gt_thresholded, task="binary")

            elif key == "grasp_width":
                grasp_width = getattr(self, f"{key}_pred")
                grasp_width_gt = getattr(self, f"{key}_gt")
                self.results["grasp_width_rmse"] = M.mean_absolute_error(grasp_width, grasp_width_gt)
                
        return self.results
