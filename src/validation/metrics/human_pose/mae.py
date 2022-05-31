from moai.monads.utils.common import dim_list

import torch
import numpy
import os

# from moai.validation.metrics.human_pose.temp import (
#     save_ply_from_keypoints
# )


class MAE(torch.nn.Module):
    def __init__(self):
        super(MAE, self).__init__()
        self.counter_m = 0
        self.counter_j = 0
        self.per_joint_results = numpy.zeros([0, 19])
        self.per_marker_results = numpy.zeros([0, 53])
        # if not os.path.exists("ply"):
        #     os.mkdir("ply")
        # if not os.path.exists("csv"):
        #     os.mkdir("csv")

    def forward(self,
        gt:         torch.Tensor,
        pred:       torch.Tensor,
    ) -> torch.Tensor:
        euc = torch.norm(gt - pred, p=2, dim=-1, keepdim=False)

        if gt.size()[1] == 53:
            self.counter_m += gt.size()[0]   
            self.per_marker_results = numpy.vstack([self.per_marker_results, euc.cpu().numpy()])
            # numpy.savetxt("csv/mae_markers.csv", self.per_marker_results, delimiter=',')
        else:
            gt[:, 14, :] = 0.0
            gt[:, 18, :] = 0.0
            pred[:, 14, :] = 0.0
            pred[:, 18, :] = 0.0
            self.counter_j += gt.size()[0]     
            self.per_joint_results = numpy.vstack([self.per_joint_results, euc.cpu().numpy()])
            # numpy.savetxt("csv/mae_joints.csv", self.per_joint_results, delimiter=',')
        
        return torch.mean(torch.mean(euc, dim=-1))
       