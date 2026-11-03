import torch.nn as nn


class QuantModel(nn.Module):
    def __init__(self, model, weight_quant_params, act_quant_params, sm_abit):
        super().__init__()
        self.model = model

    def forward(self, x, t):
        return self.model(x, t) * 1.02                  # a proportional error, so K > 0
