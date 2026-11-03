"""SDXL with SVDQuant (DeepCompressor) as the base and TCEC on top.

Run from third_party/deepcompressor/examples/diffusion, with this folder on PYTHONPATH.

  python sdxl_tcec.py collect <configs...>       official calibration data, DDIM sampling
  python sdxl_tcec.py ptq <configs...>           official quantize/evaluate, DDIM sampling
  python sdxl_tcec.py calibrate-k <configs...> --load-from MODEL   writes K to $TCEC_K

Set TCEC_MODE=paper and TCEC_K=path to make `ptq` evaluate with the correction.
All other flags are DeepCompressor's own.
"""
import os, runpy, sys
import torch
import yaml

from sdxl_scheduler import make_scheduler
from tcec_core import KStats

MODE = os.environ.get("TCEC_MODE", "none")
K_PATH = os.environ.get("TCEC_K", "")
K_N = int(os.environ.get("TCEC_K_N", 1024))           # 1024 COCO prompts, paper Sec. 4.1
K_BS = int(os.environ.get("TCEC_K_BS", 8))
PROMPTS = os.environ.get("TCEC_PROMPTS", "prompts/qdiff.yaml")


def factory(name, path, dtype, device, shift_activations):
    from deepcompressor.app.diffusion.pipeline.config import DiffusionPipelineConfig
    p = DiffusionPipelineConfig._default_build(name=name, path=path, dtype=dtype, device=device,
                                               shift_activations=shift_activations)
    K = torch.load(K_PATH)["K"] if MODE != "none" else None
    p.scheduler = make_scheduler(p.scheduler.config, K, MODE)     # SDXL defaults to Euler; the paper uses DDIM
    return p


def register():
    from deepcompressor.app.diffusion.pipeline.config import DiffusionPipelineConfig
    DiffusionPipelineConfig.register_pipeline_factory("sdxl", factory, overwrite=True)


def attach_stats(pipe, fp_unet, stats):
    """mu_q and mu from the same x_t: both networks see the pipeline's own UNet inputs"""
    q_forward, held = pipe.unet.forward, {}

    def forward(*a, **k):
        held["f"] = fp_unet(*a, **k)[0]
        return q_forward(*a, **k)

    def on_step(i, mu_q):
        f = held["f"]
        if pipe.do_classifier_free_guidance:
            u, c = f.chunk(2)
            f = u + pipe.guidance_scale * (c - u)
        stats.update(i, mu_q, f)

    pipe.unet.forward = forward
    pipe.scheduler.on_step = on_step


def calibrate_k():
    from deepcompressor.app.diffusion.config import DiffusionPtqRunConfig
    from deepcompressor.app.diffusion.nn.struct import DiffusionModelStruct
    from deepcompressor.app.diffusion.ptq import ptq

    config, *_ = DiffusionPtqRunConfig.get_parser().parse_known_args()
    assert config.load_from, "calibrate-k needs --load-from with the saved quantized model"
    fp_unet = config.pipeline.build().unet.eval()
    pipe = config.pipeline.build()
    ptq(DiffusionModelStruct.construct(pipe), config.quant, load_dirpath=config.load_from)
    kw = config.eval.get_pipeline_kwargs()
    stats = KStats(kw["num_inference_steps"], pipe.unet.config.out_channels)
    attach_stats(pipe, fp_unet, stats)
    prompts = list(yaml.safe_load(open(PROMPTS)).values())[:K_N]
    assert len(prompts) == K_N, f"{PROMPTS} has only {len(prompts)} prompts"
    pipe.set_progress_bar_config(disable=True)
    for b in range(0, K_N, K_BS):
        batch = prompts[b:b + K_BS]
        gens = [torch.Generator(device=pipe.device).manual_seed(100000 + b + j) for j in range(len(batch))]
        with torch.no_grad():
            pipe(batch, generator=gens, output_type="latent", **kw)
        print(f"{b + len(batch)}/{K_N}", flush=True)
    K = stats.solve()
    torch.save({"K": K, "steps": kw["num_inference_steps"]}, K_PATH)
    print(f"saved {K_PATH}: K mean {K.mean():+.4f} min {K.min():+.4f} max {K.max():+.4f}")


if __name__ == "__main__":
    cmd = sys.argv.pop(1)
    if cmd != "calibrate-k":
        register()
        mod = {"collect": "deepcompressor.app.diffusion.dataset.collect.calib",
               "ptq": "deepcompressor.app.diffusion.ptq"}[cmd]
        runpy.run_module(mod, run_name="__main__", alter_sys=True)
    else:
        assert K_PATH, "set TCEC_K"
        register()
        calibrate_k()
