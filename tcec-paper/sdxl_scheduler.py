"""DDIM with the TCEC correction. The UNet, quantized or not, is untouched."""
import math
from diffusers import DDIMScheduler

from tcec_core import Corrector


class TCECDDIMScheduler(DDIMScheduler):
    """model_output is the guided noise, which is the mu_q of the paper.

    on_step(i, model_output), if set, sees every step; K calibration uses it.
    """
    tcec = None
    on_step = None

    def set_timesteps(self, *a, **k):
        super().set_timesteps(*a, **k)
        self._i = 0
        if self.tcec is not None:
            self.tcec.reset()

    def _coeffs(self, t):
        t = int(t)
        ac = self.alphas_cumprod
        pt = t - self.config.num_train_timesteps // self.num_inference_steps
        at = float(ac[t])
        an = float(ac[pt]) if pt >= 0 else float(self.final_alpha_cumprod)
        return math.sqrt(1 - an) - math.sqrt(an * (1 - at) / at), math.sqrt(at / an)

    def step(self, model_output, timestep, sample, *a, return_dict=True, **k):
        out = super().step(model_output, timestep, sample, *a, return_dict=True, **k)
        if self.on_step is not None:
            self.on_step(self._i, model_output)
        self._i += 1
        if self.tcec is not None:
            B, w = self._coeffs(timestep)
            out.prev_sample = self.tcec.apply(out.prev_sample, model_output, B, w)
        return out if return_dict else (out.prev_sample, out.pred_original_sample)


def make_scheduler(config, K=None, mode="none"):
    s = TCECDDIMScheduler.from_config(config)
    if mode != "none":
        s.tcec = Corrector(K, mode)
    return s
