import torch
import typing

__all__ = [
    "FuseCoords",
]

class FuseCoords(torch.nn.Module):
    def __init__(self,
        mode:       str="two" # two, four
    ):
        super(FuseCoords,self).__init__()
        self.mode = mode

    def rotate_back_from_back(self, coords):
        coords[..., 0] = 1.0 - coords[..., 0]
        coords[..., 2] = 1.0 - coords[..., 2]
        return coords

    def rotate_left_from_right(self, coords):
        rot = torch.tensor([[
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ]]).to(coords.device).float()
        xformed_t = rot @ coords.permute(0, 2, 1)
        xformed_t += torch.tensor([0.0, 0.0, 1.0]).to(coords.device).expand(1, xformed_t.size()[2], xformed_t.size()[1]).permute(0, 2, 1)
        return xformed_t.permute(0, 2, 1)
    
    def rotate_right_from_left(self, coords):
        rot = torch.tensor([[
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ]]).to(coords.device).float()
        xformed_t = rot @ coords.permute(0, 2, 1)
        xformed_t += torch.tensor([1.0, 0.0, 0.0]).to(coords.device).expand(1, xformed_t.size()[2], xformed_t.size()[1]).permute(0, 2, 1)
        return xformed_t.permute(0, 2, 1)    

    def forward(self, coords: typing.List[torch.Tensor]) -> torch.Tensor:
        fused_coords = torch.zeros_like(coords[0])
        for i, coords_i in enumerate(coords):
            if self.mode == "two":
                if i == 0:
                    fused_coords += coords_i
                else:
                    fused_coords += self.rotate_back_from_back(coords_i)
            elif self.mode == "four":
                if i == 0:
                    fused_coords += coords_i
                elif i == 1:
                    fused_coords += self.rotate_back_from_back(coords_i)
                elif i == 2:
                    fused_coords += self.rotate_right_from_left(coords_i)
                elif i == 3:
                    fused_coords += self.rotate_left_from_right(coords_i)
        fused_coords /= len(coords)
        return fused_coords