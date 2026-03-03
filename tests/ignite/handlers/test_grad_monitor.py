import math
import torch
from ignite.engine import Engine, Events
from ignite.handlers.grad_monitor import GradMonitor


def test_grad_monitor_spike_detection():
    model = torch.nn.Linear(10, 1)
    monitor = GradMonitor(model, k=2.0)
    engine = Engine(lambda e, b: None)
    monitor.attach(engine)

    # 1.)Simulating stable training with varying gradients so std > 0.
    stable_scales = [0.08, 0.12, 0.09, 0.11, 0.10, 0.08, 0.12, 0.09, 0.11, 0.10]
    norms = []
    for scale in stable_scales:
        model.weight.grad = torch.ones_like(model.weight) * scale
        engine.run(range(1), max_epochs=1)
        norms.append(math.sqrt(10 * scale ** 2))
        assert not engine.state.unhealthy_spike

    # 2.)Simulating a sudden spike well above mean + k*std.
    model.weight.grad = torch.ones_like(model.weight) * 50.0
    engine.run(range(1), max_epochs=1)
    assert engine.state.unhealthy_spike


def test_grad_monitor_k_sensitivity():
    """A value just above mean + 1*std should spike for k=1 but not k=3."""

    def _run_monitor(k, spike_scale):
        model = torch.nn.Linear(10, 1)
        monitor = GradMonitor(model, k=k)
        engine = Engine(lambda e, b: None)
        monitor.attach(engine)

        for scale in [0.08, 0.12, 0.09, 0.11, 0.10] * 2:
            model.weight.grad = torch.ones_like(model.weight) * scale
            engine.run(range(1), max_epochs=1)

        model.weight.grad = torch.ones_like(model.weight) * spike_scale
        engine.run(range(1), max_epochs=1)
        return engine.state.unhealthy_spike

    # A moderate spike should be caught with tight k but not with loose k.
    assert _run_monitor(k=1.0, spike_scale=0.5)
    assert not _run_monitor(k=50.0, spike_scale=0.5)