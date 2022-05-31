import os

from moai.monads.utils.common import dim_list

import torch
import numpy


class RMSE(torch.nn.Module):
    def __init__(self):
        super(RMSE, self).__init__()
        self.counter_m = 0
        self.counter_j = 0
        self.per_joint_results = numpy.zeros([0, 19])
        self.per_marker_results = numpy.zeros([0, 53])
        if not os.path.exists("csv"):
            os.mkdir("csv")

    def forward(self,
        gt:         torch.Tensor,
        pred:       torch.Tensor
    ) -> torch.Tensor:
        diff_sq = torch.norm(gt - pred, p=2, dim=-1, keepdim=False) ** 2   

        if gt.size()[1] == 53:
            self.per_marker_results = numpy.vstack([self.per_marker_results, diff_sq.cpu().numpy()])
            numpy.savetxt("csv/rmse_markers.csv", self.per_marker_results, delimiter=',')
        else: 
            self.per_joint_results = numpy.vstack([self.per_joint_results, diff_sq.cpu().numpy()])
            numpy.savetxt("csv/rmse_joints.csv", self.per_joint_results, delimiter=',')

        return torch.mean(torch.sqrt(torch.mean(diff_sq, dim=-1)))