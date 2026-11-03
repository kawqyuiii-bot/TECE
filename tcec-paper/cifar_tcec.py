"""TCEC on the official Q-Diffusion CIFAR-10 base (paper Appendix J, Table 9).

Model, quantizers and checkpoints are Q-Diffusion's, unchanged. This file only adds
calibration data generation, K calibration and a sampler with the Eq. 18 correction.

  gen-cali     FP DDIM trajectories for Q-Diffusion's own calibration
  calibrate-k  K on the quantized trajectory
  sample       write PNGs for FID. none and paper share noise per image index
"""
import argparse, math, os, sys, types
import numpy as np
import torch
import yaml
from torchvision.utils import save_image

from tcec_core import Corrector, KStats, MODES

dev = "cuda" if torch.cuda.is_available() else "cpu"


def namespace(d):
    return types.SimpleNamespace(**{k: namespace(v) if isinstance(v, dict) else v for k, v in d.items()})


def setup(args):
    sys.path.insert(0, os.path.abspath(args.qdiff_root))
    from ddim.models.diffusion import Model
    from ddim.functions.ckpt_util import get_ckpt_path

    config = namespace(yaml.safe_load(open(os.path.join(args.qdiff_root, "configs/cifar10.yml"))))
    config.split_shortcut = True                                     # same as --split
    d = config.diffusion
    betas = torch.tensor(np.linspace(d.beta_start, d.beta_end, d.num_diffusion_timesteps), dtype=torch.float, device=dev)
    model = Model(config)
    model.load_state_dict(torch.load(get_ckpt_path("ema_cifar10"), map_location="cpu"))
    return model.to(dev).eval(), betas


def quantized(model, args):
    """Same construction as scripts/sample_diffusion_ddim.py with --resume"""
    from qdiff import QuantModel
    from qdiff.utils import resume_cali_model

    wq = {"n_bits": args.w_bits, "channel_wise": True, "scale_method": "max"}
    aq = {"n_bits": args.a_bits, "symmetric": True, "channel_wise": False, "scale_method": "max", "leaf_param": True}
    qnn = QuantModel(model=model, weight_quant_params=wq, act_quant_params=aq, sm_abit=args.sm_abit).to(dev).eval()
    cali = (torch.randn(1, 3, 32, 32), torch.randint(0, 1000, (1,)))
    resume_cali_model(qnn, args.cali_ckpt, cali, True, "qdiff", cond=False)
    return qnn.to(dev).eval()


def quad_seq(steps, T=1000):
    return [int(s) for s in np.linspace(0, np.sqrt(T * 0.8), steps) ** 2]


def schedule(betas, steps):
    """(t, abar_t, abar_{t-1}, B_t), last-to-first as in ddim's generalized_steps with eta = 0"""
    from ddim.functions.denoising import compute_alpha

    seq = quad_seq(steps)
    out = []
    for i, j in zip(reversed(seq), reversed([-1] + seq[:-1])):
        at = compute_alpha(betas, torch.tensor([i], device=dev)).item()
        an = compute_alpha(betas, torch.tensor([j], device=dev)).item()
        out.append((i, at, an, math.sqrt(1 - an) - math.sqrt(an * (1 - at) / at)))
    return out


def ddim(x, e, at, an):
    x0 = (x - e * math.sqrt(1 - at)) / math.sqrt(at)
    return math.sqrt(an) * x0 + math.sqrt(1 - an) * e


def noise(n, seed):
    return torch.randn(n, 3, 32, 32, generator=torch.Generator().manual_seed(seed)).to(dev)


def batches(n, bs):
    return [(b, min(bs, n - b)) for b in range(0, n, bs)]


@torch.no_grad()
def gen_cali(args):
    model, betas = setup(args)
    sc = schedule(betas, args.steps)
    xs = [[] for _ in sc]
    for b, m in batches(args.n, args.bs):
        x = noise(m, args.seed + b)
        for i, (t, at, an, _) in enumerate(sc):
            xs[i].append(x.cpu())
            x = ddim(x, model(x, torch.full((m,), float(t), device=dev)), at, an)
    data = {"xs": [torch.cat(v) for v in xs], "ts": [torch.full((args.n,), s[0], dtype=torch.long) for s in sc]}
    torch.save(data, args.out)
    print("saved", args.out, data["xs"][0].shape, "steps", len(sc))


@torch.no_grad()
def calibrate_k(args):
    model, betas = setup(args)
    sc = schedule(betas, args.steps)
    fp = setup(args)[0]                                              # QuantModel takes over `model`
    qnn = quantized(model, args)
    stats = KStats(len(sc), 3)
    for b, m in batches(args.n, args.bs):
        x = noise(m, args.seed + b)
        for i, (t, at, an, _) in enumerate(sc):
            tt = torch.full((m,), float(t), device=dev)
            e_q = qnn(x, tt)
            stats.update(i, e_q, fp(x, tt))
            x = ddim(x, e_q, at, an)
        print(f"{b + m}/{args.n}", flush=True)
    K = stats.solve()
    torch.save({"K": K, "steps": args.steps, "w_bits": args.w_bits, "a_bits": args.a_bits}, args.out)
    print(f"saved {args.out}: K mean {K.mean():+.4f} min {K.min():+.4f} max {K.max():+.4f}")


@torch.no_grad()
def sample(args):
    model, betas = setup(args)
    sc = schedule(betas, args.steps)
    net = model if args.mode == "fp" else quantized(model, args)
    K = None
    if args.mode in ("local", "paper"):
        ck = torch.load(args.k)
        assert ck["steps"] == args.steps, "K was calibrated for another step count"
        K = ck["K"]
    tc = Corrector(K, "none" if args.mode == "fp" else args.mode)
    os.makedirs(args.outdir, exist_ok=True)
    for b, m in batches(args.n, args.bs):
        if os.path.exists(os.path.join(args.outdir, f"{b + m - 1:06d}.png")):
            continue                                                 # resume after an interruption
        x = noise(m, args.seed + b)
        tc.reset()
        for i, (t, at, an, B) in enumerate(sc):
            e = net(x, torch.full((m,), float(t), device=dev))
            x = tc.apply(ddim(x, e, at, an), e, B, math.sqrt(at / an))
        x = ((x + 1) / 2).clamp(0, 1)
        for j in range(m):
            save_image(x[j], os.path.join(args.outdir, f"{b + j:06d}.png"))
        print(f"{b + m}/{args.n}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen-cali", "calibrate-k", "sample"])
    ap.add_argument("--qdiff_root", default="third_party/q-diffusion")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--w_bits", type=int, default=4)
    ap.add_argument("--a_bits", type=int, default=8)
    ap.add_argument("--sm_abit", type=int, default=8)
    ap.add_argument("--cali_ckpt", default="")
    ap.add_argument("--k", default="")
    ap.add_argument("--mode", default="none", choices=("fp",) + MODES)
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="")
    ap.add_argument("--outdir", default="")
    args = ap.parse_args()
    {"gen-cali": gen_cali, "calibrate-k": calibrate_k, "sample": sample}[args.cmd](args)
