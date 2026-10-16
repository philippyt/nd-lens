from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]

def load_config(path: str | Path = ROOT / "config.yaml") -> dict:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    for key in ("norsk_asr_dir", "checkpoint", "data_dir", "out_dir"):
        p = Path(cfg[key])
        cfg[key] = p if p.is_absolute() else (path.parent / p).resolve()
    return cfg
