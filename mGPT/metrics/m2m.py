from typing import List

import torch
from torch import Tensor
from torchmetrics import Metric

from .utils import *


# motion reconstruction metric
class PredMetrics(Metric):
    """
        A TorchMetrics class for evaluating motion prediction and in-betweening tasks.

        This class assesses how accurately a model can predict future motion frames
        (forecasting) or fill in missing frames within a sequence (in-betweening).
        It computes the error between the model's predicted motion (`joints_rst`)
        and the ground truth motion (`joints_ref`).

        The specific portion of the motion sequence being evaluated depends on the `task`
        parameter:
        - 'pred': Evaluates the final part of the sequence for motion forecasting.
        - 'inbetween': Evaluates a middle segment of the sequence for motion interpolation.

        Key Metrics Computed:
        - ADE (Average Displacement Error): Calculates the average Euclidean distance
          between predicted and ground truth joints over the evaluated time steps.
          It measures the average prediction error across the sequence.
        - FDE (Final Displacement Error): Calculates the Euclidean distance between
          the predicted and ground truth joints at the very last time step of the
          evaluated segment. It measures the error at the end point of the prediction.

        Input (to the `update` method):
        - `joints_rst` (Tensor): The predicted motion sequences from the model.
        - `joints_ref` (Tensor): The ground truth motion sequences.
        - `lengths` (List[int]): The lengths of the motion sequences in the batch.

        Output (from the `compute` method):
        - A dictionary containing the computed values for 'ADE' and 'FDE'.
    """

    def __init__(self,
                 cfg,
                 njoints: int = 22,
                 jointstype: str = "mmm",
                 force_in_meter: bool = True,
                 align_root: bool = True,
                 dist_sync_on_step=True,
                 task: str = "pred",
                 **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.name = 'Motion Prdiction'
        self.cfg = cfg
        self.jointstype = jointstype
        self.align_root = align_root
        self.task = task
        self.force_in_meter = force_in_meter

        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count_seq",
                       default=torch.tensor(0),
                       dist_reduce_fx="sum")
        # average pairwise distance
        self.add_state("APD",
                       default=torch.tensor([0.0]),
                       dist_reduce_fx="sum")
        # average displacement error
        self.add_state("ADE",
                       default=torch.tensor([0.0]),
                       dist_reduce_fx="sum")
        # final displacement error
        self.add_state("FDE",
                       default=torch.tensor([0.0]),
                       dist_reduce_fx="sum")

        self.MR_metrics = ["APD", "ADE", "FDE"]

        # All metric
        self.metrics = self.MR_metrics

    def compute(self, sanity_flag):

        count = self.count
        count_seq = self.count_seq
        mr_metrics = {}
        mr_metrics["APD"] = self.APD / count_seq
        mr_metrics["ADE"] = self.ADE / count_seq
        mr_metrics["FDE"] = self.FDE / count_seq
        
        # Reset
        self.reset()
        
        return mr_metrics

    def update(self, joints_rst: Tensor, joints_ref: Tensor,
               lengths: List[int]):
        
        assert joints_rst.shape == joints_ref.shape
        assert joints_rst.dim() == 4
        # (bs, seq, njoint=22, 3)

        self.count += sum(lengths)
        self.count_seq += len(lengths)

        rst = torch.flatten(joints_rst, start_dim=2)
        ref = torch.flatten(joints_ref, start_dim=2)
        
        for i, l in enumerate(lengths):
            if self.task == "pred":
                pred_start = int(l*self.cfg.ABLATION.predict_ratio)
                diff = rst[i,pred_start:] - ref[i,pred_start:]
            elif self.task == "inbetween":
                inbetween_start = int(l*self.cfg.ABLATION.inbetween_ratio)
                inbetween_end = l - int(l*self.cfg.ABLATION.inbetween_ratio)
                diff = rst[i,inbetween_start:inbetween_end] - ref[i,inbetween_start:inbetween_end]
            else:
                print(f"Task {self.task} not implemented.")
                diff = rst - ref
            
            dist = torch.linalg.norm(diff, dim=-1)[None]

            ade = dist.mean(dim=1)
            fde = dist[:,-1]
            self.ADE = self.ADE + ade
            self.FDE = self.FDE + fde
