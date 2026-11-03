"""IS, FID and sFID with the ADM evaluator (guided-diffusion/evaluations/evaluator.py).

torch-fidelity has no sFID. The evaluator needs TensorFlow, so it runs in its own venv.
Reference statistics are computed from the 50,000 CIFAR-10 training images.
"""
import argparse, glob, os, subprocess
import numpy as np
from PIL import Image


def pack(folder, out):
    files = sorted(glob.glob(os.path.join(folder, "*.png")))
    assert files, f"no PNGs in {folder}"
    arr = np.stack([np.asarray(Image.open(f).convert("RGB")) for f in files]).astype(np.uint8)
    np.savez(out, arr_0=arr)                                         # NHWC in [0, 255]
    return arr.shape


def cifar_ref(out, root):
    from torchvision.datasets import CIFAR10
    data = CIFAR10(root=root, train=True, download=True).data
    assert data.shape == (50000, 32, 32, 3)
    np.savez(out, arr_0=data)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True, help="folder of PNGs")
    ap.add_argument("--ref", default="runs/cifar/ref_cifar10_train.npz")
    ap.add_argument("--data_root", default="runs/cifar/data")
    ap.add_argument("--evaluator", default="third_party/guided-diffusion/evaluations/evaluator.py")
    ap.add_argument("--python", default=".venv-adm/bin/python")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if not os.path.exists(args.ref):
        cifar_ref(args.ref, args.data_root)
    npz = args.samples.rstrip("/") + ".npz"
    print("packed", pack(args.samples, npz), flush=True)
    r = subprocess.run([args.python, args.evaluator, args.ref, npz], check=True, capture_output=True, text=True)
    metrics = [l for l in r.stdout.splitlines() if l.startswith(("Inception Score", "FID", "sFID", "Precision", "Recall"))]
    assert any(l.startswith("sFID") for l in metrics), r.stdout + r.stderr
    open(args.out, "w").write("\n".join(metrics) + "\n")
    print("\n".join(metrics))
