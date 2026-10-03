"""Hard-routing restoration system (Task 2).

A classifier C looks at the input image and predicts its condition:
0 clean, 1 salt_pepper, 2 blur, 3 occlusion. The prediction picks ONE branch:

    p = softmax(C(x)),   r = argmax_k p_k

    x_hat = x               if r = clean      (identity bypass, no network)
          = A_salt(x)       if r = salt_pepper
          = A_blur(x)       if r = blur
          = A_occlusion(x)  if r = occlusion

Two evaluation modes:
  oracle routing:    the TRUE label chooses the branch (upper bound that
                     shows how good the experts are on their own job).
  predicted routing: the classifier's argmax chooses the branch (the real
                     system, which also pays for classifier mistakes).

Images are float tensors of shape (B, 3, H, W) with values in [0, 1].
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Name of the branch used for each route index (index = route = condition label).
EXPERT_NAMES = ["identity", "salt_pepper", "blur", "occlusion"]

ROUTING_MODES = ["oracle", "predicted"]


class HardRoutedRestorer(nn.Module):
    """Classifier + three specialist autoencoders with an identity bypass.

    classifier:       any module mapping (B, 3, H, W) -> logits (B, 4).
    salt_expert:      restores salt-and-pepper images (route 1).
    blur_expert:      restores blurred images (route 2).
    occlusion_expert: restores occluded images (route 3).
    Route 0 (clean) has no expert: the input is returned unchanged.
    """

    def __init__(self, classifier: nn.Module, salt_expert: nn.Module,
                 blur_expert: nn.Module, occlusion_expert: nn.Module):
        super().__init__()
        self.classifier = classifier
        # ModuleList position k-1 holds the expert for route k (1, 2, 3).
        self.experts = nn.ModuleList([salt_expert, blur_expert, occlusion_expert])

    def route(self, x: torch.Tensor):
        """Classify the batch. Returns (routes (B,) long, probs (B, 4))."""
        logits = self.classifier(x)            # (B, 4)
        probs = torch.softmax(logits, dim=1)   # each row sums to 1
        routes = probs.argmax(dim=1)           # (B,) index of the largest probability
        return routes, probs

    def restore(self, x: torch.Tensor, routes: torch.Tensor) -> torch.Tensor:
        """Send every image through the branch given by `routes` (B,).

        Starting from a copy of x means route-0 (clean) images are already
        "restored" by the identity bypass and never touch a network. Each
        expert only sees the images routed to it, and is skipped entirely
        when no image is routed to it.
        """
        routes = routes.to(x.device)
        x_hat = x.clone()
        for k, expert in enumerate(self.experts, start=1):
            mask = routes == k                 # (B,) True for images sent to expert k
            if not mask.any():
                continue                       # nobody routed here: do not run the expert
            # Run the expert on the selected images and write them back in place.
            x_hat[mask] = expert(x[mask]).to(x_hat.dtype)
        return x_hat

    def forward(self, x: torch.Tensor, routes: torch.Tensor | None = None):
        """Returns (x_hat, routes used, classifier probs).

        routes=None  -> predicted routing (classifier argmax).
        routes given -> oracle routing (the given routes are used, but the
                        classifier probs are still computed so they can be
                        reported).
        """
        predicted_routes, probs = self.route(x)
        if routes is None:
            routes = predicted_routes
        routes = routes.to(x.device).long()
        x_hat = self.restore(x, routes)
        return x_hat, routes, probs


def make_restore_fn(router: HardRoutedRestorer, mode: str, entries: list | None = None):
    """Build a `restore_fn(corrupted_batch)` for `evaluate_restoration`.

    mode "predicted": the classifier chooses the expert; `entries` is ignored.
    mode "oracle":    the true label of each image chooses the expert.

    `evaluate_restoration` only passes the corrupted batch, not the labels.
    For oracle mode the returned function therefore keeps a position
    counter and, on each call, takes the next len(batch) labels from
    `entries` (the dataset's manifest entries, in order). This is only
    correct because the evaluation loader does NOT shuffle, so batches
    arrive in the same order as `entries`. The counter is never reset:
    make a NEW restore_fn for every evaluation pass.
    """
    if mode not in ROUTING_MODES:
        raise ValueError(f"Unknown routing mode: {mode!r}. Use one of {ROUTING_MODES}.")

    if mode == "predicted":
        def restore_predicted(x: torch.Tensor) -> torch.Tensor:
            return router(x)[0]
        return restore_predicted

    if entries is None:
        raise ValueError("Oracle routing needs the manifest `entries` to know the true labels.")

    labels = torch.tensor([entry["label"] for entry in entries], dtype=torch.long)
    position = [0]  # a list so the inner function can change it

    def restore_oracle(x: torch.Tensor) -> torch.Tensor:
        start = position[0]
        end = start + len(x)
        if end > len(labels):
            raise RuntimeError("Oracle restore_fn ran past the end of `entries`; "
                               "make a new restore_fn for every evaluation pass.")
        position[0] = end
        routes = labels[start:end].to(x.device)
        return router(x, routes)[0]

    return restore_oracle


def routing_records(router: HardRoutedRestorer, dataset, device, batch_size: int = 128) -> dict:
    """Classify every item of `dataset` once, in order.

    Returns NumPy arrays aligned with `dataset.entries`:
      true  (N,)   : true condition label
      pred  (N,)   : predicted route (classifier argmax)
      probs (N, 4) : classifier softmax probabilities
    Images with true != pred are the misrouted ones.
    The caller must put the router in eval mode first.
    """
    # shuffle=False keeps the records aligned with dataset.entries.
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    true, pred, probs = [], [], []

    with torch.no_grad():
        for corrupted, _, labels in loader:
            routes, batch_probs = router.route(corrupted.to(device))
            true.append(torch.as_tensor(labels).long().cpu())
            pred.append(routes.cpu())
            probs.append(batch_probs.cpu())

    return {
        "true": torch.cat(true).numpy(),
        "pred": torch.cat(pred).numpy(),
        "probs": torch.cat(probs).numpy(),
    }
