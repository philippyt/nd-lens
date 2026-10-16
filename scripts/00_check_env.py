import torch

def main():
    print("torch", torch.__version__)
    print("torch.cuda.is_available():", torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise SystemExit("no GPU visible to torch")
    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(dev)
    print(f"device: {props.name}, {props.total_memory / 2**30:.1f} GiB")
    a = torch.randn(1024, 1024)
    b = torch.randn(1024, 1024)
    ref = (a.double() @ b.double()).float()
    for dt in (torch.float32, torch.bfloat16):
        out = (a.to(dev, dt) @ b.to(dev, dt)).float().cpu()
        err = ((out - ref).norm() / ref.norm()).item()
        print(f"{str(dt):15s} matmul rel. error {err:.2e}")
    torch.cuda.synchronize()
    print("OK")

if __name__ == "__main__":
    main()
