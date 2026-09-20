"""Baseline — EEGNet with a dense readout, trained end-to-end (GPU-ready).

Same device pattern as the other tracks' ``EEGNet`` baselines, but this
track wants a *sequence* out, not one label per window. EEGNet's head is a
convolution spanning the whole remaining time axis, which the final squeeze
then drops; ``final_conv_length=1`` slides that head instead, so the net
emits one joint-angle vector per remaining time step — ``(B, n_joints, T')``
— and the objective nearest-resamples ``T'`` up to the target's ``T``.

Angles are regressed in a standardized space: the scale is measured on the
train split in ``fit`` and kept as buffers, so a submitted state dict
restores it along with the weights.
"""

import torch
from torch import nn
from braindecode.models import EEGNet

from benchmark_utils.base_solver import CompetSolver
from benchmark_utils.data import resample_labels


class DensePose(nn.Module):
    """EEGNet emitting a joint-angle sequence, in degrees."""

    def __init__(self, n_chans, n_joints, n_times):
        super().__init__()
        self.net = EEGNet(
            n_chans=n_chans, n_outputs=n_joints, n_times=n_times,
            final_conv_length=1,
        )
        # Target scale, learned in fit; identity until then so an untrained
        # net still predicts in degrees.
        self.register_buffer("y_mean", torch.zeros(1))
        self.register_buffer("y_std", torch.ones(1))

    def forward(self, X):
        out = self.net(X)                       # standardized (B, J, T')
        # A window no longer than the net's temporal pooling leaves a
        # singleton time axis, which EEGNet's head squeezes away; keep it so
        # the output stays a sequence and the objective resamples the time
        # axis rather than the joints.
        return out[..., None] if out.ndim == 2 else out

    def to_degrees(self, out):
        return out * self.y_std + self.y_mean


class EEGNetPose:
    """Torch regressor (``fit``/``predict``) that owns its device."""

    def __init__(self, n_chans, n_joints, n_times, device,
                 lr=1e-3, n_epochs=20):
        self.device = device
        self.n_epochs = n_epochs
        self.lr = lr
        self.net = DensePose(n_chans, n_joints, n_times).to(device)

    def fit(self, train_loader):
        # One pass to set the target scale, so the loss is well conditioned
        # whatever the angle range of the study.
        total = count = 0.0
        sq = 0.0
        for _X, y, _info in train_loader:
            y = y.float()
            total += y.sum()
            sq += (y ** 2).sum()
            count += y.numel()
        mean = total / count
        self.net.y_mean.fill_(float(mean))
        self.net.y_std.fill_(float((sq / count - mean ** 2).clamp_min(1e-8)
                                   .sqrt()))

        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        self.net.train()
        for _ in range(self.n_epochs):
            for X, y, _info in train_loader:
                # X, y already live on self.device (moved by the loader).
                opt.zero_grad()
                pred = self.net(X)                       # (B, J, T')
                # Compare at the net's resolution; the objective resamples
                # predictions back up at evaluation time.
                target = resample_labels(y.float(), pred.shape[-1])
                target = (target - self.net.y_mean) / self.net.y_std
                loss = loss_fn(pred, target)
                loss.backward()
                opt.step()
        return self

    @torch.no_grad()
    def predict(self, X):
        self.net.eval()
        X = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        return self.net.to_degrees(self.net(X))   # (B, J, T') degrees


class Solver(CompetSolver):

    name = "EEGNet"

    requirements = ["pip::braindecode"]

    def load_model(self, meta):
        model = EEGNetPose(
            n_chans=meta["n_chans"],
            n_joints=meta["n_joints"],
            n_times=meta["n_times"],
            device=self.device,
        )
        # A submitted network ships the state dict written by ``save_model``;
        # without it (a local training run) the net starts from scratch.
        weights = meta["submission_dir"] / "weights.pt"
        if weights.exists():
            model.net.load_state_dict(
                torch.load(weights, map_location=self.device))
        return model

    # Optional training helpers
    def fit(self, model, train_loader):
        model.fit(train_loader)

    def save_model(self, model, path):
        torch.save(model.net.state_dict(), path / "weights.pt")
