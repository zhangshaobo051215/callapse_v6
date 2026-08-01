from src.model import VisionTransformer
from src.param_groups import classify_parameters


def test_hidden_policy_exact():
    model = VisionTransformer({})
    controlled, uncontrolled = classify_parameters(model)
    assert len(controlled) == 48
    assert not any(x.startswith(("head.", "patch_embed.")) or "norm" in x or x.endswith("bias")
                   for x in controlled)
    assert set(controlled).isdisjoint(uncontrolled)
    assert len(controlled) + len(uncontrolled) == len(list(model.parameters()))

