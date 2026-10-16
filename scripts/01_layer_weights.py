import json
from collections import Counter

import numpy as np
from safetensors.numpy import load_file

from dialect_probing.config import load_config

def main():
    cfg = load_config()
    ck = cfg["checkpoint"]
    state = load_file(ck / "model.safetensors")
    meta = json.loads((ck / "config.json").read_text())
    print("checkpoint config:", json.dumps(meta["config"]))

    w = state["layer_weights"].astype(np.float64)
    p = np.exp(w - w.max())
    p /= p.sum()
    print("\nstate  logit    weight")
    for i, (logit, prob) in enumerate(zip(w, p)):
        name = "input (after subsampling)" if i == 0 else f"layer {i}"
        print(f"{i:>5}  {logit:+.3f}  {prob:.4f}  {'#' * round(prob * 200)}  {name}")
    print(f"sum={p.sum():.4f}  uniform would be {1 / len(p):.4f}")

    enc = Counter(k.split(".")[2] for k in state if k.startswith("encoder.layers."))
    other_enc = sorted({k for k in state if k.startswith("encoder.") and not k.startswith("encoder.layers.")})
    print("\nencoder layer indices overridden by checkpoint (0-based):", dict(enc))
    print("other encoder tensors in checkpoint:", other_enc[:10])
    out = cfg["out_dir"] / "layer_weights.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"logits": w.tolist(), "weights": p.tolist()}, indent=1))
    print("wrote", out)

if __name__ == "__main__":
    main()
