"""Wide ResNets on CIFAR-10: depth/width and dropout study, MixUp fine-tuning, CIFAR-10-C robustness.

The notebook `lab1_wide_resnet_cifar10.ipynb` drives this module:
    prepare(...)                      downloads CIFAR-10 and CIFAR-10-C and builds the fixed splits
    python wrn_lab.py worker ...      one process per GPU; takes jobs from the shared list until none is left

Every job writes one JSON file to <out>/results/, so a run can stop and resume without losing finished work.
"""
import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------- settings

CORRUPTIONS = {   # the 15 corruptions of the CIFAR-10-C benchmark, in its four groups
    "noise": ["gaussian_noise", "shot_noise", "impulse_noise"],
    "blur": ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur"],
    "weather": ["snow", "frost", "fog", "brightness"],
    "digital": ["contrast", "elastic_transform", "pixelate", "jpeg_compression"],
}
CIFAR10C_URL = "https://zenodo.org/records/2535967/files/CIFAR-10-C.tar"
CIFAR10C_MD5 = "56bf5dcef84df0e2308c6dcbcbbd8499"
CIFAR10C_BYTES = 2918471680
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def make_plan(smoke=False):
    """All experiment settings in one place. `smoke` is a tiny version used only to test the code on a CPU."""
    plan = dict(
        configs=[[40, 4], [16, 8], [28, 10]],   # depth, widening factor: the three best WRNs of Table 5 in the paper
        dropouts=[0.0, 0.3],                    # 0.3 is the paper's dropout rate for CIFAR
        seeds=[0, 1, 2],
        split_seed=1234,
        n_train_per_class=2000,                 # 20,000 training images (40% of the training set)
        n_val_per_class=500,                    # 5,000 validation images, also from the training set
        epochs=15, batch=128, lr=0.1, momentum=0.9, weight_decay=5e-4,
        ft_epochs=5, ft_lr=0.01, mixup_alpha=0.2,
        c_per_severity=2000,                    # CIFAR-10-C images per corruption and severity (same images everywhere)
        corruptions=CORRUPTIONS,
        eval_batch=1000,
        smoke=False,
    )
    if smoke:
        plan.update(configs=[[10, 1], [10, 2]], seeds=[0, 1], n_train_per_class=24, n_val_per_class=8,
                    epochs=2, batch=32, ft_epochs=1, c_per_severity=16, eval_batch=64, smoke=True,
                    corruptions={g: v[:2] for g, v in CORRUPTIONS.items()})
    return plan


def corruption_names(plan):
    return [c for group in plan["corruptions"].values() for c in group]


def sweep_jobs(plan):
    """Parts 1 and 2: every architecture with and without dropout, for each seed (largest models first)."""
    jobs = []
    for seed in plan["seeds"]:
        for depth, k in sorted(plan["configs"], key=lambda c: -c[0] * c[1] ** 2):
            for p in plan["dropouts"]:
                jobs.append(dict(kind="sweep", depth=depth, k=k, dropout=p, seed=seed,
                                 name=f"wrn{depth}-{k}_drop{p:.1f}_seed{seed}"))
    return jobs


def mixup_jobs(plan, best):
    """Part 3 and the bonus: the chosen model before and after 5 more epochs, without and with MixUp."""
    jobs = []
    for seed in plan["seeds"]:
        for kind in ("base", "nomixup", "mixup"):
            jobs.append(dict(kind=kind, depth=best["depth"], k=best["k"], dropout=best["dropout"], seed=seed,
                             name=f"{kind}_seed{seed}"))
    return jobs


# ----------------------------------------------------------------------------- model

class Block(nn.Module):
    """Pre-activation basic block of a wide ResNet: BN-ReLU-conv3x3, dropout, BN-ReLU-conv3x3."""

    def __init__(self, c_in, c_out, stride, dropout):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(c_in)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.dropout = dropout
        self.shortcut = None if (c_in == c_out and stride == 1) else nn.Conv2d(c_in, c_out, 1, stride, 0, bias=False)

    def forward(self, x):
        o = F.relu(self.bn1(x))
        y = self.conv1(o)
        y = F.relu(self.bn2(y))
        if self.dropout > 0:
            y = F.dropout(y, p=self.dropout, training=self.training)
        y = self.conv2(y)
        return y + (x if self.shortcut is None else self.shortcut(o))


class WideResNet(nn.Module):
    """WRN-depth-k (Zagoruyko and Komodakis, 2016): 3 groups of (depth-4)/6 blocks, widths 16k, 32k, 64k."""

    def __init__(self, depth, k, dropout=0.0, num_classes=10):
        super().__init__()
        assert (depth - 4) % 6 == 0, "depth must be 6n+4"
        n = (depth - 4) // 6
        widths = [16, 16 * k, 32 * k, 64 * k]
        self.conv = nn.Conv2d(3, widths[0], 3, 1, 1, bias=False)
        blocks = []
        for g, stride in enumerate((1, 2, 2)):
            for b in range(n):
                blocks.append(Block(widths[g] if b == 0 else widths[g + 1], widths[g + 1],
                                    stride if b == 0 else 1, dropout))
        self.blocks = nn.Sequential(*blocks)
        self.bn = nn.BatchNorm2d(widths[3])
        self.fc = nn.Linear(widths[3], num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.relu(self.bn(self.blocks(self.conv(x))))
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ----------------------------------------------------------------------------- data

def _md5_download(url, dest, md5, size=None, tries=3):
    for attempt in range(1, tries + 1):
        h, done, t0 = hashlib.md5(), 0, time.time()
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as f:
                while True:
                    chunk = r.read(1 << 22)
                    if not chunk:
                        break
                    f.write(chunk)
                    h.update(chunk)
                    done += len(chunk)
                    if size and done % (1 << 29) < (1 << 22):
                        print(f"  {done / 1e9:.1f} of {size / 1e9:.1f} GB ({time.time() - t0:.0f} s)", flush=True)
            if h.hexdigest() == md5:
                return
            print(f"  checksum mismatch on attempt {attempt}", flush=True)
        except OSError as e:
            print(f"  download error on attempt {attempt}: {e}", flush=True)
    raise RuntimeError(f"could not download {url} with the expected checksum")


def _fake_images(rng, labels):
    """Smoke test only: random images whose average colour depends on the class, so a model can learn a little."""
    base = np.linspace(40, 215, 10)
    x = rng.normal(base[labels][:, None, None, None], 35, size=(len(labels), 32, 32, 3))
    return x.clip(0, 255).astype(np.uint8)


def prepare(plan, data_dir):
    """Build <data>/cifar10_split.npz and <data>/cifar10c_subset.npz once (skipped when they already exist)."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    split_file, c_file = data_dir / "cifar10_split.npz", data_dir / "cifar10c_subset.npz"
    rng = np.random.default_rng(plan["split_seed"])
    names = corruption_names(plan)

    if not split_file.exists():
        if plan["smoke"]:
            y_all = np.repeat(np.arange(10), 60)
            x_all = _fake_images(rng, y_all)
            y_test = np.repeat(np.arange(10), 20)
            x_test = _fake_images(rng, y_test)
        else:
            from torchvision.datasets import CIFAR10
            tr = CIFAR10(str(data_dir / "cifar10"), train=True, download=True)
            te = CIFAR10(str(data_dir / "cifar10"), train=False, download=True)
            x_all, y_all = tr.data, np.array(tr.targets)
            x_test, y_test = te.data, np.array(te.targets)
        tr_idx, va_idx = [], []
        for c in range(10):   # class-balanced, disjoint training and validation sets from the official training set
            idx = rng.permutation(np.flatnonzero(y_all == c))
            va_idx.append(idx[:plan["n_val_per_class"]])
            tr_idx.append(idx[plan["n_val_per_class"]:plan["n_val_per_class"] + plan["n_train_per_class"]])
        tr_idx, va_idx = np.sort(np.concatenate(tr_idx)), np.sort(np.concatenate(va_idx))
        x_train = x_all[tr_idx]
        mean = x_train.reshape(-1, 3).mean(0) / 255
        std = x_train.reshape(-1, 3).std(0) / 255
        np.savez(split_file, x_train=x_train, y_train=y_all[tr_idx], x_val=x_all[va_idx], y_val=y_all[va_idx],
                 x_test=x_test, y_test=y_test, mean=mean, std=std, train_index=tr_idx, val_index=va_idx)
        shutil.rmtree(data_dir / "cifar10", ignore_errors=True)

    if not c_file.exists():
        n_test = 10000 if not plan["smoke"] else 200
        pick = np.sort(rng.choice(n_test, plan["c_per_severity"], replace=False))
        x_c = np.empty((len(names), 5, len(pick), 32, 32, 3), np.uint8)
        if plan["smoke"]:
            y_test = np.load(split_file)["y_test"]
            y_c = y_test[pick]
            for i in range(len(names)):
                for s in range(5):
                    x_c[i, s] = _fake_images(np.random.default_rng(100 * i + s), y_c)
        else:
            tar_path, raw = data_dir / "CIFAR-10-C.tar", data_dir / "CIFAR-10-C_raw"
            print(f"Downloading CIFAR-10-C ({CIFAR10C_BYTES / 1e9:.1f} GB) from Zenodo ...", flush=True)
            _md5_download(CIFAR10C_URL, tar_path, CIFAR10C_MD5, CIFAR10C_BYTES)
            wanted = {f"{n}.npy" for n in names + ["labels"]}
            raw.mkdir(exist_ok=True)
            with tarfile.open(tar_path) as tar:
                for m in tar.getmembers():
                    base = m.name.rsplit("/", 1)[-1]
                    if m.isfile() and base in wanted:
                        with tar.extractfile(m) as src, open(raw / base, "wb") as dst:
                            shutil.copyfileobj(src, dst, 1 << 22)
            missing = wanted - {p.name for p in raw.iterdir()}
            assert not missing, f"not found in the CIFAR-10-C archive: {sorted(missing)}"
            tar_path.unlink()
            labels = np.load(raw / "labels.npy")
            y_c = labels[pick]
            for s in range(5):   # the labels repeat for each of the 5 severities
                assert np.array_equal(labels[s * 10000 + pick], y_c)
            assert np.array_equal(y_c, np.load(split_file)["y_test"][pick])
            for i, n in enumerate(names):
                arr = np.load(raw / f"{n}.npy", mmap_mode="r")
                for s in range(5):
                    x_c[i, s] = arr[s * 10000 + pick]
                del arr   # release the memory map before the file is deleted
            shutil.rmtree(raw)
        np.savez(c_file, x=x_c, y=y_c, names=np.array(names), test_index=pick)

    d = np.load(split_file)
    return {"train": len(d["y_train"]), "val": len(d["y_val"]), "test": len(d["y_test"]),
            "cifar10c": f"{len(names)} corruptions x 5 severities x {plan['c_per_severity']} images"}


class Data:
    """Keeps the image arrays on the device as uint8 and turns them into normalized float batches on demand."""

    def __init__(self, data_dir, device):
        d = np.load(Path(data_dir) / "cifar10_split.npz")
        self.device = device
        to = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)
        self.x = {s: to(d[f"x_{s}"]) for s in ("train", "val", "test")}
        self.y = {s: to(d[f"y_{s}"]).long() for s in ("train", "val", "test")}
        self.mean = torch.tensor(d["mean"], dtype=torch.float32, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(d["std"], dtype=torch.float32, device=device).view(1, 3, 1, 1)
        self._c = None
        self.data_dir = Path(data_dir)

    def to_float(self, xb):
        x = xb.permute(0, 3, 1, 2).float().div_(255)
        return ((x - self.mean) / self.std).contiguous(memory_format=torch.channels_last)

    @property
    def corrupted(self):
        if self._c is None:
            c = np.load(self.data_dir / "cifar10c_subset.npz")
            self._c = dict(x=c["x"], y=torch.from_numpy(c["y"]).long().to(self.device), names=list(c["names"]))
        return self._c


def augment(x, g):
    """Random 32x32 crop of the image padded by 4 pixels with reflections, then a random horizontal flip."""
    b, dev = x.shape[0], x.device
    xp = F.pad(x, (4, 4, 4, 4), mode="reflect")
    i = torch.randint(0, 9, (b,), device=dev, generator=g)
    j = torch.randint(0, 9, (b,), device=dev, generator=g)
    ar = torch.arange(32, device=dev)
    rows = (i[:, None] + ar)[:, :, None]
    cols = (j[:, None] + ar)[:, None, :]
    out = xp[torch.arange(b, device=dev)[:, None, None], :, rows, cols].permute(0, 3, 1, 2)
    flip = torch.rand(b, device=dev, generator=g) < 0.5
    out = torch.where(flip[:, None, None, None], out.flip(3), out)
    return out.contiguous(memory_format=torch.channels_last)


# ----------------------------------------------------------------------------- training and evaluation

def _autocast(device):
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=str(device).startswith("cuda"))


def _scaler(device):
    on = str(device).startswith("cuda")
    try:
        return torch.amp.GradScaler("cuda", enabled=on)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=on)


@torch.no_grad()
def predict(model, data, xs, ys, batch):
    """Softmax probabilities (float32, on the CPU) and the mean cross-entropy for uint8 images xs with labels ys."""
    model.eval()
    probs, loss_sum = [], 0.0
    for s in range(0, len(xs), batch):
        xb = xs[s:s + batch]
        if not torch.is_tensor(xb):
            xb = torch.from_numpy(np.ascontiguousarray(xb)).to(data.device)
        with _autocast(data.device):
            logits = model(data.to_float(xb)).float()
        loss_sum += F.cross_entropy(logits, ys[s:s + batch], reduction="sum").item()
        probs.append(F.softmax(logits, 1).cpu())
    return torch.cat(probs), loss_sum / len(xs)


def train(model, data, plan, seed, epochs, lr0, mixup_alpha=0.0, log=print, tag=""):
    """SGD with Nesterov momentum and a cosine learning rate from lr0 to 0; validation after every epoch."""
    dev = data.device
    torch.manual_seed(seed)
    g_shuffle = torch.Generator().manual_seed(seed)
    g_aug = torch.Generator(device=dev).manual_seed(seed + 10_000)
    np_rng = np.random.default_rng(seed + 20_000)
    opt = torch.optim.SGD(model.parameters(), lr=lr0, momentum=plan["momentum"], nesterov=True,
                          weight_decay=plan["weight_decay"])
    scaler = _scaler(dev)
    n, bs = len(data.y["train"]), plan["batch"]
    steps = math.ceil(n / bs)
    total, step = epochs * steps, 0
    hist = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc", "seconds")}
    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        perm = torch.randperm(n, generator=g_shuffle).to(dev)
        loss_sum = correct = 0.0
        for s in range(0, n, bs):
            lr = 0.5 * lr0 * (1 + math.cos(math.pi * step / total))
            for group in opt.param_groups:
                group["lr"] = lr
            idx = perm[s:s + bs]
            x = augment(data.to_float(data.x["train"][idx]), g_aug)
            y = data.y["train"][idx]
            with _autocast(dev):
                if mixup_alpha > 0:   # MixUp (Zhang et al., 2018): blend each image with another one of the batch
                    lam = float(np_rng.beta(mixup_alpha, mixup_alpha))
                    other = torch.randperm(len(y), device=dev, generator=g_aug)
                    out = model(lam * x + (1 - lam) * x[other])
                    loss = lam * F.cross_entropy(out, y) + (1 - lam) * F.cross_entropy(out, y[other])
                else:
                    out = model(x)
                    loss = F.cross_entropy(out, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            pred = out.argmax(1)
            if mixup_alpha > 0:
                hits = lam * (pred == y).sum() + (1 - lam) * (pred == y[other]).sum()
            else:
                hits = (pred == y).sum()
            loss_sum += loss.item() * len(y)
            correct += float(hits)
            step += 1
        probs, val_loss = predict(model, data, data.x["val"], data.y["val"], plan["eval_batch"])
        val_acc = (probs.argmax(1) == data.y["val"].cpu()).float().mean().item()
        hist["train_loss"].append(loss_sum / n)
        hist["train_acc"].append(correct / n)
        hist["val_loss"].append(val_loss)
        hist["val_acc"].append(val_acc)
        hist["seconds"].append(time.time() - t0)
        log(f"{tag} epoch {epoch + 1}/{epochs}: train loss {loss_sum / n:.3f}, train acc {correct / n:.3f}, "
            f"val acc {val_acc:.4f} ({time.time() - t0:.0f} s, {n / (time.time() - t0):.0f} img/s)")
    return hist


def confidence_stats(probs, labels, n_bins=15, n_hist=20):
    """Confidence = largest softmax probability. Histograms for right and wrong answers, and calibration (ECE)."""
    conf, pred = probs.max(1)
    right = (pred == labels).numpy()
    conf = conf.numpy().astype(np.float64)
    edges = np.linspace(0, 1, n_bins + 1)
    which = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    bin_acc, bin_conf, bin_count = [], [], []
    ece = 0.0
    for b in range(n_bins):
        m = which == b
        cnt = int(m.sum())
        bin_count.append(cnt)
        bin_acc.append(float(right[m].mean()) if cnt else None)
        bin_conf.append(float(conf[m].mean()) if cnt else None)
        if cnt:
            ece += cnt / len(conf) * abs(right[m].mean() - conf[m].mean())
    h_edges = np.linspace(0, 1, n_hist + 1)
    return dict(acc=float(right.mean()), mean_conf=float(conf.mean()),
                mean_conf_right=float(conf[right].mean()) if right.any() else None,
                mean_conf_wrong=float(conf[~right].mean()) if (~right).any() else None,
                ece=float(ece), bin_acc=bin_acc, bin_conf=bin_conf, bin_count=bin_count,
                hist_right=np.histogram(conf[right], h_edges)[0].tolist(),
                hist_wrong=np.histogram(conf[~right], h_edges)[0].tolist(), n=int(len(conf)))


def evaluate_robustness(model, data, plan):
    """Clean test set and every CIFAR-10-C corruption and severity."""
    probs, test_loss = predict(model, data, data.x["test"], data.y["test"], plan["eval_batch"])
    clean = confidence_stats(probs, data.y["test"].cpu())
    clean["loss"] = test_loss
    c = data.corrupted
    acc = np.zeros((len(c["names"]), 5))
    all_probs = []
    for i in range(len(c["names"])):
        for s in range(5):
            p, _ = predict(model, data, c["x"][i, s], c["y"], plan["eval_batch"])
            acc[i, s] = (p.argmax(1) == c["y"].cpu()).float().mean().item()
            all_probs.append(p)
    corrupted = confidence_stats(torch.cat(all_probs), c["y"].cpu().repeat(len(c["names"]) * 5))
    return dict(clean=clean, cifar10c=dict(names=c["names"], acc=acc.tolist(), mean_acc=float(acc.mean()),
                                           confidence=corrupted))


# ----------------------------------------------------------------------------- jobs and workers

def run_job(job, plan, data, ckpt_dir, log):
    t0 = time.time()
    model = None
    torch.manual_seed(job["seed"])
    model = WideResNet(job["depth"], job["k"], job["dropout"]).to(data.device).to(memory_format=torch.channels_last)
    res = dict(job=job, params=count_params(model))
    base_ckpt = Path(ckpt_dir) / f"wrn{job['depth']}-{job['k']}_drop{job['dropout']:.1f}_seed{job['seed']}.pt"
    if job["kind"] == "sweep":
        res["history"] = train(model, data, plan, job["seed"], plan["epochs"], plan["lr"], log=log, tag=job["name"])
        probs, test_loss = predict(model, data, data.x["test"], data.y["test"], plan["eval_batch"])
        res["val_acc"] = res["history"]["val_acc"][-1]
        res["test_acc"] = (probs.argmax(1) == data.y["test"].cpu()).float().mean().item()
        res["test_loss"] = test_loss
        torch.save(model.state_dict(), base_ckpt)
    else:
        model.load_state_dict(torch.load(base_ckpt, map_location=data.device, weights_only=True))
        if job["kind"] != "base":
            alpha = plan["mixup_alpha"] if job["kind"] == "mixup" else 0.0
            res["history"] = train(model, data, plan, job["seed"] + 100, plan["ft_epochs"], plan["ft_lr"],
                                   mixup_alpha=alpha, log=log, tag=job["name"])
        res.update(evaluate_robustness(model, data, plan))
    res["seconds"] = time.time() - t0
    res["device"] = torch.cuda.get_device_name(data.device) if str(data.device).startswith("cuda") else "cpu"
    return res


def worker(phase, device, data_dir, out_dir, ckpt_dir, deadline):
    out_dir, ckpt_dir = Path(out_dir), Path(ckpt_dir)
    plan = json.loads((out_dir / "plan.json").read_text())
    if phase == "sweep":
        jobs = sweep_jobs(plan)
    else:
        jobs = mixup_jobs(plan, json.loads((out_dir / "best.json").read_text()))
    for d in ("results", "locks", "logs"):
        (out_dir / d).mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tag = device.replace(":", "")
    logf = open(out_dir / "logs" / f"{phase}_{tag}.log", "a", encoding="utf8")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')} {tag}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    if device.startswith("cuda"):
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
    data = Data(data_dir, device)
    for job in jobs:
        result_file = out_dir / "results" / f"{job['name']}.json"
        if result_file.exists():
            continue
        try:
            fd = os.open(out_dir / "locks" / f"{job['name']}.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
        except FileExistsError:
            continue
        if time.time() > deadline:
            log(f"time limit reached, not starting {job['name']}")
            os.remove(out_dir / "locks" / f"{job['name']}.lock")
            break
        log(f"start {job['name']}")
        res = run_job(job, plan, data, ckpt_dir, log)
        tmp = result_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(res))
        tmp.replace(result_file)
        log(f"done {job['name']} in {res['seconds'] / 60:.1f} min")
    logf.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("worker")
    w.add_argument("--phase", choices=["sweep", "mixup"], required=True)
    w.add_argument("--device", required=True)
    w.add_argument("--data", required=True)
    w.add_argument("--out", required=True)
    w.add_argument("--ckpt", required=True)
    w.add_argument("--deadline", type=float, default=float("inf"))
    a = ap.parse_args()
    worker(a.phase, a.device, a.data, a.out, a.ckpt, a.deadline)
    sys.exit(0)
