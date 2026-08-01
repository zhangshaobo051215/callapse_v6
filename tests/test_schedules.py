import math

from src.schedules import base_lr_at, schedule_ratio


KW = dict(control_start_step=100, total_steps=300, cyclic_period_steps=80)


def test_ratios_and_period():
    assert all(schedule_ratio(n, 100, **KW) == 1 for n in
               ("constant", "linear_up", "linear_down", "cyclic"))
    assert schedule_ratio("linear_up", 300, **KW) == 2
    assert math.isclose(schedule_ratio("linear_down", 300, **KW), 1 / 3)
    assert math.isclose(schedule_ratio("cyclic", 100, **KW),
                        schedule_ratio("cyclic", 180, **KW))
    values = [schedule_ratio("cyclic", s, **KW) for s in range(100, 181)]
    assert min(values) >= .5 and max(values) <= 1.5


def test_base_lr_endpoints():
    kw = dict(peak_lr=6e-4, final_lr=6e-5, warmup_steps=20,
              decay_start_step=270, total_steps=300)
    assert base_lr_at(0, **kw) == 6e-4 / 20
    assert base_lr_at(19, **kw) == 6e-4
    assert base_lr_at(20, **kw) == 6e-4
    assert base_lr_at(270, **kw) == 6e-5 + (6e-4 - 6e-5)
    assert base_lr_at(299, **kw) > 6e-5

