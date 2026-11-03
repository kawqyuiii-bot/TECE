import os, tempfile, torch
from ddim.models.diffusion import Model


def get_ckpt_path(name):
    p = os.path.join(tempfile.gettempdir(), "fake_ema.pt")
    torch.save(Model(None).state_dict(), p)
    return p
