import torch
from ignite.engine import Engine, Events

class GradMonitor:
    """
    Monitors L2 gradient norm to catch training instability early using dynamic thresholds.
    The threshold is defined as mean + (k * std), calculated using Welford's algorithm 
    for memory efficiency.

    Args:
        model: The model to monitor.
        k: Multiplier for standard deviation to define the spike threshold (default: 3.0).

    Example:
        .. code-block:: python

            monitor = GradMonitor(model)
            monitor.attach(trainer)

            # Defining a custom train step that respects the spike flag.
            def custom_train_step(engine, batch):
                # GradMonitor has already run because it's attached to ITERATION_STARTED
                if getattr(engine.state, "unhealthy_spike", False):
                    # Skip the entire forward/backward pass for this batch
                    return {"skipped": True}
                
                # Normal training logic.
                model.train()
                optimizer.zero_grad()
                # ... forward/backward/step ...
                return {"loss": loss.item()}

            trainer = Engine(custom_train_step)

    .. versionadded:: 0.6.0
    """

    def __init__(self, model: torch.nn.Module, k: float = 3.0):
        self.model = model
        self.k = k
        self.device = None
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0 

    def __call__(self, engine: Engine):
        if self.device is None:
            self.device = next(self.model.parameters()).device

        total_norm_sq = torch.tensor(0.0, device=self.device)
        
        for p in self.model.parameters():
            if p.grad is not None:
                # Optimized: pow(2).sum() is faster than norm()**2.
                total_norm_sq += p.grad.pow(2).sum()

        total_norm = torch.sqrt(total_norm_sq).item()
        
        scaler = getattr(engine.state, "scaler", None)
        if scaler is not None:
            total_norm /= scaler.get_scale()

        if self.count > 1:
            std = (self.m2 / (self.count - 1)) ** 0.5
            threshold = self.mean + (self.k * std)
            engine.state.unhealthy_spike = total_norm > threshold
        else:
            engine.state.unhealthy_spike = False

        self.count += 1
        delta = total_norm - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (total_norm - self.mean)

    def attach(self, engine: Engine):
        engine.add_event_handler(Events.ITERATION_STARTED, self)