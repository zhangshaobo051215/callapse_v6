import copy

import torch
from torch.nn import functional as F

from src.model import ViTConfig, VisionTransformer

CFG = ViTConfig(image_size=16, embed_dim=12, depth=1, num_heads=3, num_classes=4)

def _continue(state, opt_state, batches):
    model = VisionTransformer(CFG)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    model.load_state_dict(state); opt.load_state_dict(opt_state)
    losses = []
    for x, y in batches:
        opt.zero_grad(); loss = F.cross_entropy(model(x), y); loss.backward(); opt.step()
        losses.append(loss.item())
    return losses, model.state_dict()


def test_resume_determinism():
    torch.manual_seed(7)
    model = VisionTransformer(CFG)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    state, opt_state = copy.deepcopy(model.state_dict()), copy.deepcopy(opt.state_dict())
    batches = [(torch.randn(2, 3, 16, 16), torch.randint(0, 4, (2,))) for _ in range(20)]
    a_loss, a = _continue(state, opt_state, batches)
    b_loss, b = _continue(state, opt_state, batches)
    assert a_loss == b_loss
    assert max((a[k] - b[k]).abs().max().item() for k in a) < 1e-7
