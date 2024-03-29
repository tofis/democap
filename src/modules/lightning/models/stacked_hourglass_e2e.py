import moai.networks.lightning as minet
import moai.nn.convolution as mic
import moai.nn.residual as mires
import moai.nn.sampling.spatial.downsample as mids
import moai.modules.lightning as mimod
import moai.nn.utils as miu

import torch

import hydra.utils as hyu
import omegaconf.omegaconf as omegaconf
import typing
import logging

log = logging.getLogger(__name__)

#NOTE: from https://github.com/anibali/pytorch-stacked-hourglass/blob/master/src/stacked_hourglass/model.py
#NOTE: from https://github.com/princeton-vl/pytorch_stacked_hourglass/blob/master/models/posenet.py

__all__ = ["StackedHourglassMod_e2e"]

class StackedHourglassMod_e2e(torch.nn.Module):
    def __init__(self,
        configuration:  omegaconf.DictConfig,
        modules:        omegaconf.DictConfig,
        # data:           omegaconf.DictConfig=None,
        # parameters:     omegaconf.DictConfig=None,
        # feedforward:    omegaconf.DictConfig=None,
        # monads:         omegaconf.DictConfig=None,
        # supervision:    omegaconf.DictConfig=None,
        # validation:     omegaconf.DictConfig=None,
        # visualization:  omegaconf.DictConfig=None,
        # export:         omegaconf.DictConfig=None,
    ):
        super(StackedHourglassMod_e2e, self).__init__(
            # data=data, parameters=parameters,
            # feedforward=feedforward, monads=monads,
            # supervision=supervision, validation=validation,
            # export=export, visualization=visualization,            
        )        
        self.stacks = configuration.stacks
        preproc = configuration.preproc
        projection = configuration.projection
        prediction = configuration.prediction
        merge = configuration.merge
        hourglass = list(modules.values())[0]
        self.out = configuration.out_features
        if not preproc == None: 
            self.pre = torch.nn.Sequential(
                mic.make_conv_block(
                    block_type=preproc.block,
                    convolution_type=preproc.convolution,
                    in_features=configuration.in_features, 
                    out_features=hourglass.features // 4,
                    activation_type=preproc.activation,
                    convolution_params={
                        "kernel_size": preproc.stem.kernel_size,
                        "stride": preproc.stem.stride,
                        "padding": preproc.stem.padding,
                    },
                ),
                mires.make_residual_block(
                    block_type=preproc.residual,
                    convolution_type=preproc.convolution,
                    in_features=hourglass.features // 4,
                    out_features=hourglass.features // 2,
                    bottleneck_features=hourglass.features // 2,
                    activation_type=preproc.activation,
                    strided=False,
                ),
                mids.make_downsample(
                    downscale_type=preproc.downscale,
                    features=hourglass.features // 2,
                    kernel_size=3 if preproc.downscale == 'maxpool2d_aa' else 2,
                ),
                mires.make_residual_block(
                    block_type=preproc.residual,
                    convolution_type=preproc.convolution,
                    in_features=hourglass.features // 2,
                    out_features=hourglass.features // 2,
                    bottleneck_features=hourglass.features // 2,
                    activation_type=preproc.activation,
                    strided=False,
                ),
                mires.make_residual_block(
                    block_type=preproc.residual,
                    convolution_type=preproc.convolution,
                    in_features=hourglass.features // 2,
                    out_features=hourglass.features,
                    bottleneck_features=hourglass.features,
                    activation_type=preproc.activation,
                    strided=False,
                ),
            )
        
        self.hgs = torch.nn.ModuleList([
                torch.nn.Sequential(
                    hyu.instantiate(hourglass)
                ) for i in range(self.stacks)
            ] 
        )
        
        if not preproc == None: 
            self.features = torch.nn.ModuleList([
                    torch.nn.Sequential(
                        mires.make_residual_block(
                            block_type=preproc.residual,
                            convolution_type=preproc.convolution,
                            in_features=hourglass.features,
                            out_features=hourglass.features,
                            bottleneck_features=hourglass.features,
                            activation_type=preproc.activation,
                            strided=False,
                        ),
                        mic.make_conv_block(
                            block_type=projection.block,
                            convolution_type=projection.convolution,
                            in_features=hourglass.features, 
                            out_features=hourglass.features,
                            activation_type=projection.activation,
                            convolution_params={"kernel_size": 1, "padding": 0},
                        )
                    ) for i in range(self.stacks)
                ]
            )
        else:
            self.features = torch.nn.ModuleList([
                    torch.nn.Sequential(                        
                        mic.make_conv_block(
                            block_type=projection.block,
                            convolution_type=projection.convolution,
                            in_features=hourglass.features, 
                            out_features=hourglass.features,
                            activation_type=projection.activation,
                            convolution_params={"kernel_size": 1, "padding": 0},
                        )
                    ) for i in range(self.stacks)
                ]
            )
        
        self.outs = torch.nn.ModuleList([
            mic.make_conv_block(
                block_type=prediction.block,
                convolution_type=prediction.convolution,
                in_features=hourglass.features, 
                out_features=configuration.out_features,
                activation_type=prediction.activation,
                convolution_params={
                    "kernel_size": 1, "padding": 0, 
                },
                activation_params={"inplace": True} 
            ) for i in range(self.stacks)         
        ])
        self.merge_features = torch.nn.ModuleList([
            torch.nn.Sequential(
                mic.make_conv_1x1(
                    convolution_type=projection.convolution,
                    in_channels=hourglass.features,
                    out_channels=hourglass.features
                ),
                torch.nn.Dropout2d(p=merge.dropout, inplace=True)\
                    if merge.dropout > 0.0 else torch.nn.Identity()
            ) for i in range(self.stacks-1)
        ])
        self.merge_preds = torch.nn.ModuleList([
            torch.nn.Sequential(
                mic.make_conv_1x1(
                    convolution_type=projection.convolution,
                    in_channels=configuration.out_features,
                    out_channels=hourglass.features
                ),
                torch.nn.Dropout2d(p=prediction.dropout, inplace=False)\
                    if prediction.dropout > 0.0 else torch.nn.Identity()
            ) for i in range(self.stacks-1)
        ])
        # self.input = configuration.input #NOTE: cannot be as required in cascase model
        self.output_prefix = configuration.output

    def forward(self, 
        data: torch.Tensor
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = data
        if hasattr(self, 'pre'):
            x = self.pre(x)
        combined_hm_preds = []
        for i in range(self.stacks):
            hg = self.hgs[i](x)
            feature = self.features[i](hg)
            preds = self.outs[i](feature)
            combined_hm_preds.append(preds)
            if i < self.stacks - 1:                
                x = x + self.merge_preds[i](preds) + self.merge_features[i](feature)            
        
        aggregated_hm = torch.zeros_like(combined_hm_preds[0])
        for i, heatmap in enumerate(combined_hm_preds):
            aggregated_hm += heatmap

        return aggregated_hm[:, :19, ...], aggregated_hm[:, 19:, ...], torch.cat([x, aggregated_hm], dim=1)
        
        # return aggregated_hm[:, :53, ...], aggregated_hm[:, 53:, ...]

    