from moai.utils.arguments import ensure_string_list

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

#NOTE: from https://github.com/HRNet/HRNet-Bottom-Up-Pose-Estimation/blob/master/lib/models/pose_hrnet.py
#NOTE: from https://arxiv.org/pdf/1908.07919.pdf

__all__ = ["HRNetMod_e2e"]

class HRNetMod_e2e(torch.nn.Module):
    def __init__(self,
        configuration:  omegaconf.DictConfig,
        modules:        omegaconf.DictConfig
    ):
        super(HRNetMod_e2e, self).__init__(
            # data=data, parameters=parameters,
            # feedforward=feedforward, monads=monads,
            # supervision=supervision, validation=validation,
            # export=export, visualization=visualization,            
        )
        preproc = configuration.preproc
        residual = configuration.residual
        #NOTE: preproc = stem + layer1
        preproc_convs = []
        prev_features = configuration.in_features
        self.out = configuration.out_features
        if not preproc == None:
            stem = preproc.stem
            for b, c, a, f, k, s, p in zip(
                stem.blocks, stem.convolutions,
                stem.activations, stem.features,
                stem.kernel_sizes, stem.strides, stem.paddings):
                preproc_convs.append(mic.make_conv_block(
                    block_type=b,
                    convolution_type=c,
                    in_features=prev_features, 
                    out_features=f,
                    activation_type=a,
                    convolution_params={
                        "kernel_size": k,
                        "stride": s,
                        "padding": p,
                    },
                ))
                prev_features = f
            residual_blocks = []
            for i, o, b in zip(
                residual.features.in_features, residual.features.out_features,
                residual.features.bottleneck_features, 
            ):
                residual_blocks.append(mires.make_residual_block(
                    block_type=residual.block,
                    convolution_type=residual.convolution,
                    out_features=o,
                    in_features=i,
                    bottleneck_features=b,
                    activation_type=residual.activation,
                    strided=False,
                ))
            self.pre = torch.nn.Sequential(
                *preproc_convs, *residual_blocks,
            )

            start_transition_key = 'start_transition_standard_1'
            highres_key = 'highres_standard_1'
            stage_transition_key = 'stage_transition_standard_1'
            head_key = 'top_branch_1'
        else:
            start_transition_key = 'start_transition_standard_2'
            highres_key = 'highres_standard_2'
            stage_transition_key = 'stage_transition_standard_2'
            head_key = 'top_branch_2'

        branches_config = configuration.branches
        start_trans_config = modules[start_transition_key]
        self.start_trans = hyu.instantiate(start_trans_config, 
            in_features=residual.features.out_features[-1],
            start_features=branches_config.start_features
        )
        #NOTE: stages
        highres_module = modules[highres_key] # NOTE: outputs list of # branches outputs
        self.stages = torch.nn.ModuleList([
            torch.nn.Sequential(*[
                hyu.instantiate(highres_module, 
                    branches=i, depth=d, start_features=branches_config.start_features
                ) for _, d in zip(range(modules), depths)
            ]) for i, modules, depths in zip(
                range(2, configuration.stages + 1),
                branches_config.modules,
                branches_config.depths,
            )
        ])
        stage_trans_config = modules[stage_transition_key]
        self.stage_transitions = torch.nn.ModuleList([
            hyu.instantiate(stage_trans_config, branches=i + 1,
                prev_branch_features=branches_config.start_features * (2 ** i),
            ) for i in range(1, configuration.stages - 1)
        ])
        head_module = modules[head_key]
        self.head = hyu.instantiate(head_module,
            stages=configuration.stages,
            start_features=branches_config.start_features,
            out_features=configuration.out_features,
        )
        # self.input = ensure_string_list(configuration.input)
        # self.output = ensure_string_list(configuration.output)
        self.output_prefix = configuration.output

    def forward(self, 
        data: torch.Tensor
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = data
        if hasattr(self, 'pre'):
            x = self.pre(x)
        hr_inputs = self.start_trans(x)
        combined_hm_preds = []
        combined_hm_preds.append(hr_inputs)
        for stage, trans in zip(self.stages, self.stage_transitions):
            features = stage(hr_inputs)
            combined_hm_preds.append(features)
            hr_inputs = trans(features)

        combined_hm_preds.append(self.stages[-1](hr_inputs))
        combined_hm_preds_final = []
        for i, features in enumerate(combined_hm_preds):
            combined_hm_preds_final.append(self.head(features))

        aggregated_hm = torch.zeros_like(combined_hm_preds_final[0])
        for i, heatmap in enumerate(combined_hm_preds_final):
            aggregated_hm += heatmap        
        
  
        return aggregated_hm[:, :53, ...], aggregated_hm[:, 53:, ...], torch.cat([x, aggregated_hm], dim=1)
 