from moai.monads.utils import spatial_dim_list

import torch

__all__ = ["zMean"]

class zMean(torch.nn.Module):
    def __init__(self,
    ):
        super(zMean, self).__init__()
   
    def forward(self, 
        heatmaps: torch.Tensor, # spatial probability tensor of K keypoints
    ) -> torch.Tensor:
            hm_dims = spatial_dim_list(heatmaps)
            return torch.mean(heatmaps, dim=tuple(hm_dims)).unsqueeze(2)
            # return torch.amax(heatmaps, dim=tuple(hm_dims)).unsqueeze(2)
            # return torch.sum(heatmaps, dim=tuple(hm_dims)).unsqueeze(2)
