import json
import sys
import time

import torch
import torch.nn.functional as F
from norsk_asr.data import BatchLoader, PackedSet, bucket_batches
from norsk_asr.train import export_checkpoint, load_checkpoint, save_checkpoint

from dialect_probing.config import load_config

LR, WD, WARMUP, EPOCHS, BATCH_SECONDS, SEED = 5e-5, 0.01, 50, 3, 600.0, 0

def main():
    arm = sys.argv[1]
    cfg = load_config()
    root = cfg["out_dir"] / "style" / arm
    device = torch.device("cuda")
    torch.manual_seed(SEED)
    model, units = load_checkpoint(cfg["checkpoint"], device)
    model.set_trainable(encoder=False)
    train = PackedSet(sorted(root.glob("*.jsonl")), max_seconds=30)
    targets = [units.encode(r["text"]) for r in train.rows]
    keep = [i for i, t in enumerate(targets) if 0 < len(t) <= 0.9 * train.rows[i]["n"] / 640]
    train.rows, targets = [train.rows[i] for i in keep], [targets[i] for i in keep]
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"{arm}: {len(train.rows)} utterances, {sum(r['n'] for r in train.rows) / 16000 / 3600:.2f} h, "
          f"{sum(p.numel() for p in params) / 1e6:.1f}M trainable", flush=True)
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WD, betas=(0.9, 0.98), fused=True)
    model.train()
    model.encoder.eval()
    step, t0 = 0, time.time()
    for epoch in range(EPOCHS):
        loader = torch.utils.data.DataLoader(BatchLoader(train, bucket_batches(train, BATCH_SECONDS, seed=SEED + epoch), targets),
                                             batch_size=None, shuffle=False, num_workers=8, pin_memory=True)
        for b in loader:
            opt.param_groups[0]["lr"] = LR * min(1.0, (step + 1) / WARMUP)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lp, olens = model(b["wav"].to(device, non_blocking=True), b["lengths"].to(device, non_blocking=True))
            loss = F.ctc_loss(lp.transpose(0, 1), b["targets"].to(device), olens, b["target_lengths"].to(device),
                              blank=0, reduction="sum", zero_infinity=True) / b["target_lengths"].sum().to(device)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
            if torch.isfinite(gn):
                opt.step()
            step += 1
            if step % 25 == 0 or step == 1:
                print(json.dumps({"step": step, "epoch": epoch, "loss": round(loss.item(), 4), "grad_norm": round(gn.item(), 3),
                                  "minutes": round((time.time() - t0) / 60, 1)}), flush=True)
    save_checkpoint(root / "model.pt", model, units, step, {"arm": arm, "lr": LR, "epochs": EPOCHS})
    export_checkpoint(root / "model.pt", root / "run")
    (root / "model.pt").unlink()
    print(f"{arm}: {step} steps in {(time.time() - t0) / 60:.1f} min -> {root / 'run'}")

if __name__ == "__main__":
    main()
