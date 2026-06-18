# Seed: starter `submission.py`

A foundation-model submission is a benchopt solver subclassing
`CompetEEGSolver`. Copy this, swap in your model, and follow the
"Participation" page to test it locally.

```python
import torch

from benchmark_utils.base_solver import CompetEEGSolver


class Solver(CompetEEGSolver):
    # Shown on the leaderboard.
    name = "MyModel"

    # Inherit the base requirements (scikit-learn, torch) and add your own.
    requirements = CompetEEGSolver.requirements + ["pip::my-model-pkg"]

    def load_model(self, meta):
        """Load and return your *frozen* model (called once).

        ``meta`` has: sfreq, ch_names, chs_info, n_chans, n_times,
        n_classes, task.
        """
        model = load_my_pretrained_model()
        model.eval()
        model.requires_grad_(False)
        return model

    def time_embed(self, model, X):
        """Windows ``X: (B, C, T)`` (torch) -> temporal embedding (B, T', D)."""
        with torch.inference_mode():
            return model.encode(X)

    # Optional: override for custom pooling (default: mean over time of
    # ``time_embed``), used for epoched tasks.
    # def embed(self, model, X):
    #     return self.time_embed(model, X)[:, 0]   # e.g. a CLS token
```

The competition fits the scikit-learn linear head on top of your embeddings and
scores both the epoched (motor imagery) and dense (sleep) tasks. See
`solution/submission.py` in the bundle for a complete example wrapping
braindecode's REVE.
