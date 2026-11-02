"""Auditable NABirds species-level evaluation with explicit human overlap review.

Name matching only finds candidates. Absence of a name match never approves a
class; a reviewer must check taxonomy/synonyms and assign a species ancestor.
"""

import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath, PureWindowsPath

from PIL import Image
from torch.utils.data import Dataset


PROTOCOL = "nabirds-manual-nonoverlap-v1"
METADATA_FILES = ("classes.txt", "images.txt", "image_class_labels.txt", "hierarchy.txt")


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize_bird_name(name):
    """Normalize spelling for candidate discovery, without claiming taxonomy."""
    name = unicodedata.normalize("NFKD", name).casefold()
    name = re.sub(r"^\d+\.", "", name)
    name = re.sub(r"\([^)]*\)", "", name)
    name = name.replace("’", "'").replace("'", "")
    return re.sub(r"[^a-z0-9]+", "", name)


def _table(path, integer_keys=False, allow_empty=False):
    result = {}
    for number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"Invalid metadata row: {path}:{number}")
        key = int(parts[0]) if integer_keys else parts[0]
        if key in result:
            raise ValueError(f"Duplicate ID {key} in {path}")
        result[key] = parts[1].strip()
    if not result and not allow_empty:
        raise ValueError(f"Empty metadata: {path}")
    return result


def _safe_image_path(root, relative):
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError(f"Invalid image path: {relative}")
    path = PurePosixPath(relative)
    if path.is_absolute() or PureWindowsPath(relative).drive or ".." in path.parts:
        raise ValueError(f"Unsafe image path: {relative}")
    image_root = (Path(root) / "images").resolve()
    resolved = (image_root / relative).resolve()
    if not resolved.is_relative_to(image_root):
        raise ValueError(f"Image path escapes images directory: {relative}")
    return resolved


def load_nabirds_metadata(data_root):
    root = Path(data_root)
    classes = _table(root / "classes.txt", integer_keys=True)
    images = _table(root / "images.txt")  # NABirds image IDs can be UUID strings.
    labels = {key: int(value) for key, value in _table(root / "image_class_labels.txt").items()}
    parents = {key: int(value) for key, value in _table(root / "hierarchy.txt", integer_keys=True, allow_empty=True).items()}
    if images.keys() != labels.keys():
        raise ValueError("images.txt and image_class_labels.txt must contain identical image IDs")
    if set(labels.values()) - classes.keys():
        raise ValueError("Image label references an unknown class")
    if (set(parents) | set(parents.values())) - classes.keys():
        raise ValueError("Hierarchy references an unknown class")
    paths = set()
    for relative in images.values():
        resolved = _safe_image_path(root, relative)
        if resolved in paths:
            raise ValueError(f"Duplicate image path: {relative}")
        paths.add(resolved)
    for class_id in classes:
        _ancestors(class_id, parents)
    return {"classes": classes, "images": images, "labels": labels, "parents": parents,
            "hashes": {name: sha256_file(root / name) for name in METADATA_FILES}}


def _ancestors(class_id, parents):
    result = [class_id]
    while class_id in parents:
        class_id = parents[class_id]
        if class_id in result:
            raise ValueError("Cycle in NABirds hierarchy")
        result.append(class_id)
    return result


def create_review(data_root, cub_train_classes):
    """Generate a pending row for every image-bearing visual category."""
    metadata = load_nabirds_metadata(data_root)
    cub = _table(cub_train_classes, integer_keys=True)
    normalized_cub = defaultdict(list)
    for class_id, name in cub.items():
        normalized_cub[normalize_bird_name(name)].append({"class_id": class_id, "name": name})
    counts = Counter(metadata["labels"].values())
    rows = []
    for class_id, count in sorted(counts.items()):
        ancestry = _ancestors(class_id, metadata["parents"])
        candidates = []
        for node in ancestry:
            for candidate in normalized_cub[normalize_bird_name(metadata["classes"][node])]:
                match = {**candidate, "matched_nabirds_class_id": node}
                if match not in candidates:
                    candidates.append(match)
        # Default only a suggestion; reviewer must choose the species node.
        same_name = [node for node in ancestry if normalize_bird_name(metadata["classes"][node]) ==
                     normalize_bird_name(metadata["classes"][class_id])]
        rows.append({"class_id": class_id, "name": metadata["classes"][class_id],
                     "image_count": count,
                     "ancestry": [{"class_id": node, "name": metadata["classes"][node]} for node in ancestry],
                     "candidate_cub_overlap": candidates, "species_class_id": same_name[-1],
                     "decision": "pending", "reviewer": "", "evidence": ""})
    return {"protocol": PROTOCOL, "source_train_classes_sha256": sha256_file(cub_train_classes),
            "source_metadata_sha256": metadata["hashes"], "rows": rows}


def build_manifest(data_root, cub_train_classes, review_file, min_images=1):
    """Validate a review and build the selected species manifest.

    An empty/unreviewed template fails closed. All selected image files must
    exist. No image or class is selected merely because its name did not match.
    """
    if min_images < 1:
        raise ValueError("min_images must be positive")
    metadata = load_nabirds_metadata(data_root)
    expected = create_review(data_root, cub_train_classes)
    review = json.loads(Path(review_file).read_text(encoding="utf-8-sig"))
    for field in ("protocol", "source_train_classes_sha256", "source_metadata_sha256"):
        if review.get(field) != expected[field]:
            raise ValueError(f"Review does not match current metadata: {field}")
    rows = review.get("rows", [])
    row_map = {row["class_id"]: row for row in rows}
    if len(row_map) != len(rows) or set(row_map) != {row["class_id"] for row in expected["rows"]}:
        raise ValueError("Review must contain exactly one row per image-bearing category")
    approved = {}
    audit = []
    for original in expected["rows"]:
        row = row_map[original["class_id"]]
        decision = row.get("decision")
        if decision not in ("pending", "exclude", "non_overlap"):
            raise ValueError("Review decision must be pending, exclude, or non_overlap")
        for field in ("name", "image_count", "ancestry", "candidate_cub_overlap"):
            if row.get(field) != original[field]:
                raise ValueError(f"Review evidence was changed for class {row['class_id']}: {field}")
        if decision != "pending" and any(not isinstance(row.get(key), str) or not row[key].strip()
                                          for key in ("reviewer", "evidence")):
            raise ValueError("Every human decision requires reviewer and evidence")
        if decision == "non_overlap":
            if original["candidate_cub_overlap"]:
                raise ValueError("A name/hierarchy overlap candidate cannot be approved as non_overlap")
            species_id = row.get("species_class_id")
            if isinstance(species_id, bool) or not isinstance(species_id, int) or species_id not in _ancestors(row["class_id"], metadata["parents"]):
                raise ValueError("species_class_id must be the category itself or a hierarchy ancestor")
            if species_id not in metadata["parents"]:
                raise ValueError("The NABirds root category cannot be used as a species")
            approved[row["class_id"]] = species_id
        audit.append(row)
    species_ids = set(approved.values())
    for species_id in species_ids:
        if species_ids.intersection(_ancestors(species_id, metadata["parents"])[1:]):
            raise ValueError("Selected species overlap through a parent/child relationship")
        visual_descendants = {visual_id for visual_id in metadata["labels"].values()
                              if species_id in _ancestors(visual_id, metadata["parents"])}
        if any(approved.get(visual_id) != species_id for visual_id in visual_descendants):
            raise ValueError("Selected species has an unreviewed or excluded visual subclass")
    # A matching descendant can reveal an overlap missed by a generic parent name.
    for row in expected["rows"]:
        if row["candidate_cub_overlap"] and species_ids.intersection(_ancestors(row["class_id"], metadata["parents"])):
            raise ValueError("Selected species contains a descendant with a CUB overlap candidate")
    counts = Counter(approved[c] for c in metadata["labels"].values() if c in approved)
    included = {species for species, count in counts.items() if count >= min_images}
    if not included:
        raise ValueError("No human-approved non-overlapping species meet min_images")
    samples = []
    for image_id, relative in sorted(metadata["images"].items()):
        visual_id = metadata["labels"][image_id]
        species_id = approved.get(visual_id)
        if species_id not in included:
            continue
        if not _safe_image_path(data_root, relative).is_file():
            raise FileNotFoundError(f"Missing selected NABirds image: {relative}")
        samples.append({"image_id": image_id, "path": relative, "class_id": species_id,
                        "visual_class_id": visual_id})
    return {"protocol": PROTOCOL, "dataset": "nabirds", "evaluation_only": True,
            "class_unit": "human-reviewed species", "overlap_review": "human-confirmed-non-overlap",
            "source_train_classes_sha256": expected["source_train_classes_sha256"],
            "source_metadata_sha256": metadata["hashes"], "review_sha256": sha256_file(review_file),
            "min_images": min_images,
            "classes": [{"class_id": species, "name": metadata["classes"][species],
                         "image_count": counts[species]} for species in sorted(included)],
            "samples": samples, "audit": audit}


class NABirdsDataset(Dataset):
    """Image loader for reviewed manifests; verifies original metadata and audit."""

    def __init__(self, data_root, manifest_file, transform=None, *, review_file=None,
                 cub_train_classes=None):
        self.data_root = Path(data_root)
        self.transform = transform
        self.manifest_hash = sha256_file(manifest_file)
        manifest = json.loads(Path(manifest_file).read_text(encoding="utf-8-sig"))
        self.manifest_metadata = manifest

        if (review_file is None) != (cub_train_classes is None):
            raise ValueError("review_file 与 cub_train_classes 必须同时提供")
        if review_file is not None:
            if sha256_file(review_file) != manifest.get("review_sha256"):
                raise ValueError("审核记录哈希与 NABirds 清单不一致")
            rebuilt = build_manifest(data_root, cub_train_classes, review_file,
                                     manifest.get("min_images", 1))
            if rebuilt != manifest:
                raise ValueError("NABirds 清单与当前审核记录重新构建的内容不一致")
        if (manifest.get("protocol") != PROTOCOL or manifest.get("dataset") != "nabirds" or
                manifest.get("evaluation_only") is not True or
                manifest.get("overlap_review") != "human-confirmed-non-overlap"):
            raise ValueError("NABirds evaluation requires a human-reviewed non-overlap manifest")
        metadata = load_nabirds_metadata(data_root)
        if manifest.get("source_metadata_sha256") != metadata["hashes"]:
            raise ValueError("NABirds metadata changed since the manifest was built")
        for field in ("source_train_classes_sha256", "review_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", str(manifest.get(field, ""))):
                raise ValueError(f"Missing or invalid audit hash: {field}")
        classes = manifest.get("classes", [])
        class_ids = [row["class_id"] for row in classes]
        if not classes or len(set(class_ids)) != len(class_ids):
            raise ValueError("Empty or duplicated manifest species")
        for row in classes:
            if metadata["classes"].get(row["class_id"]) != row["name"]:
                raise ValueError("Manifest class does not match NABirds metadata")
        self.classes = [row["name"] for row in classes]
        self.class_to_idx = {class_id: index for index, class_id in enumerate(class_ids)}
        audit_rows = manifest.get("audit", [])
        audit = {row["class_id"]: row for row in audit_rows}
        if len(audit) != len(audit_rows) or set(audit) != set(metadata["labels"].values()):
            raise ValueError("Manifest requires the complete visual-category audit")
        self.samples = []
        self.class_to_indices = defaultdict(list)
        seen = set()
        for row in manifest.get("samples", []):
            image_id, species_id = row["image_id"], row["class_id"]
            visual_id = row["visual_class_id"]
            if image_id in seen or metadata["images"].get(image_id) != row["path"] or metadata["labels"].get(image_id) != visual_id:
                raise ValueError("Duplicated image or image metadata mismatch in manifest")
            seen.add(image_id)
            reviewed = audit.get(visual_id, {})
            if (reviewed.get("decision") != "non_overlap" or reviewed.get("species_class_id") != species_id or
                    reviewed.get("candidate_cub_overlap") != [] or
                    any(not isinstance(reviewed.get(key), str) or not reviewed[key].strip() for key in ("reviewer", "evidence")) or
                    species_id not in _ancestors(visual_id, metadata["parents"]) or species_id not in self.class_to_idx):
                raise ValueError("Manifest sample lacks an explicit valid human non-overlap review")
            image_path = _safe_image_path(data_root, row["path"])
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            label = self.class_to_idx[species_id]
            self.class_to_indices[label].append(len(self.samples))
            self.samples.append((image_path, label))
        if set(self.class_to_indices) != set(range(len(classes))):
            raise ValueError("Empty manifest class")
        for index, row in enumerate(classes):
            if len(self.class_to_indices[index]) != row["image_count"]:
                raise ValueError("Manifest image count mismatch")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as source:
            image = source.convert("RGB")
        return (self.transform(image) if self.transform else image), label


NABirdsManifestDataset = NABirdsDataset
