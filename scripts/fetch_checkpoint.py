from huggingface_hub import hf_hub_download

from dialect_probing.config import load_config

REPO = "ahdrage/norsk-asr"

def main():
    cfg = load_config()
    run = cfg["checkpoint"].name
    for f in ("model.safetensors", "config.json", "units.json", "args.json"):
        print("got", hf_hub_download(REPO, f"runs/{run}/{f}", local_dir=cfg["norsk_asr_dir"]))

if __name__ == "__main__":
    main()
