"""Extract the audited DM-Diaz SimCLR ResNet-18 encoder from a Lightning checkpoint.

Never unpickle checkpoint objects: ``torch.load`` is restricted to ``weights_only=True``.
The sequential backbone keys are translated to torchvision ResNet-18 names and
checked exhaustively before saving an encoder-only state dict.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torchvision.models import resnet18

from pattern_recognition_coursework.engine.experiment import PROJECT_ROOT, project_output


SOURCE_URL = "https://huggingface.co/DM-Diaz/SimCLR-ResNet18-ImageNet1K"
SOURCE_SHA256 = "4a50dfd4f78f83b8eba1a7c358cd0a476a9da1cf5579b7cf01a2051597ca217a"
STAGES = {"0": "conv1", "1": "bn1", "4": "layer1", "5": "layer2", "6": "layer3", "7": "layer4"}


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def extract_encoder(checkpoint: dict) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("state_dict"), dict):
        raise ValueError("需要含 state_dict 的 Lightning 检查点")
    source = checkpoint["state_dict"]
    encoder = {}
    for key, tensor in source.items():
        if not key.startswith("backbone."):
            continue
        parts = key.split(".", 2)
        if len(parts) < 3 or parts[1] not in STAGES or not isinstance(tensor, torch.Tensor):
            raise ValueError(f"不支持的编码器张量: {key}")
        translated = STAGES[parts[1]] + "." + parts[2]
        if translated in encoder:
            raise ValueError(f"重复的编码器张量: {translated}")
        encoder[translated] = tensor.cpu()
    target = resnet18(weights=None)
    target.fc = nn.Identity()
    expected = target.state_dict()
    if encoder.keys() != expected.keys():
        raise ValueError(f"编码器张量不完整: missing={sorted(expected.keys()-encoder.keys())}, unexpected={sorted(encoder.keys()-expected.keys())}")
    mismatched = [key for key, value in expected.items() if encoder[key].shape != value.shape or encoder[key].dtype != value.dtype]
    if mismatched:
        raise ValueError(f"编码器张量形状或类型不匹配: {mismatched}")
    target.load_state_dict(encoder, strict=True)
    return encoder


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "data/cache/weights/checkpoint_100-resnet18-simclr-imagenet1k.ckpt")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/cache/weights/simclr_resnet18_backbone.pt")
    parser.add_argument("--provenance", type=Path, default=PROJECT_ROOT / "reports/records/simclr_weight_provenance.json")
    args = parser.parse_args(argv)
    args.output = project_output(args.output)
    args.provenance = project_output(args.provenance)
    if args.output.exists() or args.provenance.exists():
        parser.error("目标文件已存在；拒绝覆盖")
    source_hash = digest(args.checkpoint)
    if source_hash != SOURCE_SHA256:
        parser.error(f"来源 SHA-256 不匹配: {source_hash}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    encoder = extract_encoder(checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(encoder, args.output)
    record = {
        "source": SOURCE_URL, "checkpoint_filename": args.checkpoint.name,
        "source_sha256": source_hash, "source_epoch_zero_based": checkpoint.get("epoch"),
        "source_global_step": checkpoint.get("global_step"),
        "extraction": "backbone.0/1/4/5/6/7 mapped strictly to torchvision conv1/bn1/layer1-4; projection head omitted",
        "encoder_tensors": len(encoder), "output": str(args.output), "output_sha256": digest(args.output),
        "load_policy": "torch.load(weights_only=True), strict architecture keys/shapes/dtypes",
    }
    args.provenance.parent.mkdir(parents=True, exist_ok=True)
    args.provenance.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()
