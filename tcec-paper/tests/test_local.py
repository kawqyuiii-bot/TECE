"""Tests that need neither a GPU nor the third-party repos.

The Q-Diffusion interfaces are replaced by small stand-ins in tests/fake_qdiff, so this
checks our formulas and control flow, not the official calibration.
"""
import os, subprocess, sys
import numpy as np
import torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from eval_adm import pack
from tcec_core import Corrector, KStats
from sdxl_scheduler import make_scheduler
from diffusers import DDIMScheduler


def test_k_closed_form():
    torch.manual_seed(0)
    f = torch.randn(32, 3, 8, 8)
    q = f / (1 - torch.tensor([0.05, -0.02, 0.1]).view(1, 3, 1, 1))     # (1 - K) q = f
    s = KStats(1, 3)
    s.update(0, q, f)
    K = s.solve()[0]
    assert torch.allclose(K, torch.tensor([0.05, -0.02, 0.1]), atol=2e-3), K


def test_lambda_matches_paper():
    q, f = torch.randn(4, 3, 8, 8), torch.randn(4, 3, 8, 8)
    s = KStats(1, 3)
    s.update(0, q, f)
    lam = 0.01 * q.pow(2).mean((0, 2, 3)) / f.var((0, 2, 3), unbiased=False)
    num, den = (q.pow(2) - q * f).sum((0, 2, 3)), q.pow(2).sum((0, 2, 3)) + lam + 1e-8
    assert torch.allclose(s.solve()[0], num / den, atol=1e-5)


def test_eq18_terms():
    K = torch.tensor([[.1, .1, .1], [.2, .2, .2]])
    c = Corrector(K, "paper")
    e1, e2 = torch.ones(1, 3, 2, 2), 2 * torch.ones(1, 3, 2, 2)
    x = torch.zeros(1, 3, 2, 2)
    a = c.apply(x, e1, 0.5, 9.0)                       # no t+1 term at the first step
    assert torch.allclose(a, -0.5 * .1 * e1)
    b = c.apply(x, e2, 0.25, 2.0)                      # -B_t K_t e_t - w B_{t+1} K_{t+1} e_{t+1}
    assert torch.allclose(b, -0.25 * .2 * e2 - 2.0 * 0.5 * .1 * e1)
    c.reset()
    assert c.prev is None and c.i == 0


def test_none_is_identity():
    x = torch.randn(1, 3, 2, 2)
    c = Corrector(None, "none")
    assert c.apply(x, x, 1.0, 1.0) is x


def scheduler(mode, K=None):
    base = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                         set_alpha_to_one=False, steps_offset=1, timestep_spacing="leading")
    return make_scheduler(base.config, K, mode)


def loop(s, steps=5, seed=0):
    torch.manual_seed(seed)
    net = torch.nn.Conv2d(4, 4, 3, padding=1)
    s.set_timesteps(steps)
    x = torch.randn(2, 4, 8, 8)
    for t in s.timesteps:
        with torch.no_grad():
            x = s.step(net(x), t, x, return_dict=False)[0]
    return x


def test_scheduler_none_equals_ddim():
    plain = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                          set_alpha_to_one=False, steps_offset=1, timestep_spacing="leading")
    assert torch.equal(loop(plain), loop(scheduler("none")))


def test_scheduler_coeffs_reconstruct_ddim_step():
    s = scheduler("none")
    s.set_timesteps(10)
    x, e = torch.randn(1, 4, 4, 4), torch.randn(1, 4, 4, 4)
    for t in s.timesteps:
        ac = s.alphas_cumprod
        pt = int(t) - 100
        an = ac[pt] if pt >= 0 else s.final_alpha_cumprod
        B, w = s._coeffs(t)
        a = (an / ac[int(t)]).sqrt()
        got = a * x + B * e
        assert torch.allclose(got, s.step(e, t, x).prev_sample, atol=1e-4)
        assert abs(w - 1 / float(a)) < 1e-5


def test_scheduler_resets_between_runs():
    K = torch.full((5, 4), 0.1)
    s = scheduler("paper", K)
    assert torch.equal(loop(s), loop(s))                # state left by run 1 must not reach run 2
    assert not torch.equal(loop(s), loop(scheduler("none")))


def test_cifar_cli():
    import tempfile
    d = tempfile.mkdtemp(dir=os.environ.get("TMPDIR"))
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "tests", "fake_qdiff"))
    run = lambda *a: subprocess.run([sys.executable, os.path.join(ROOT, "cifar_tcec.py"), *a,
                                     "--qdiff_root", os.path.join(ROOT, "tests", "fake_qdiff")],
                                    check=True, env=env, cwd=ROOT, capture_output=True, text=True)
    run("gen-cali", "--steps", "6", "--n", "4", "--bs", "2", "--out", f"{d}/c.pt")
    data = torch.load(f"{d}/c.pt")
    assert len(data["xs"]) == 6 and data["xs"][0].shape == (4, 3, 32, 32) and data["ts"][0][0] > data["ts"][-1][0]
    run("calibrate-k", "--steps", "6", "--n", "4", "--bs", "2", "--cali_ckpt", "x", "--out", f"{d}/K.pt")
    K = torch.load(f"{d}/K.pt")["K"]
    assert K.shape == (6, 3) and K.abs().max() > 0
    for m in ("none", "paper"):
        run("sample", "--steps", "6", "--n", "4", "--bs", "2", "--cali_ckpt", "x", "--k", f"{d}/K.pt",
            "--mode", m, "--outdir", f"{d}/{m}")
        assert len(os.listdir(f"{d}/{m}")) == 4
    from PIL import Image
    a = np.asarray(Image.open(f"{d}/none/000000.png")).astype(float)
    b = np.asarray(Image.open(f"{d}/paper/000000.png")).astype(float)
    assert 0 < np.abs(a - b).mean() < 40                # same noise, small correction


def test_adm_pack():
    import tempfile
    from PIL import Image
    d = tempfile.mkdtemp(dir=os.environ.get("TMPDIR"))
    px = np.random.default_rng(0).integers(0, 256, (3, 32, 32, 3), dtype=np.uint8)
    for i, a in enumerate(px):
        Image.fromarray(a).save(f"{d}/{i:06d}.png")
    assert pack(d, f"{d}.npz") == (3, 32, 32, 3)
    got = np.load(f"{d}.npz")["arr_0"]
    assert got.dtype == np.uint8 and np.array_equal(got, px)


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v()
            print("ok", k, flush=True)
