"""The label remapper of the training kit (no GPU, no real images needed)."""
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_remap_to_our_five_classes_and_resplit(tmp_path):
    src = tmp_path / "src"
    (src / "train" / "images").mkdir(parents=True)
    (src / "train" / "labels").mkdir(parents=True)
    yaml.safe_dump({"names": ["Swimming", "drowning", "person", "ball"]}, open(src / "data.yaml", "w"))
    for i in range(10):
        (src / "train" / "images" / f"im{i}.jpg").write_bytes(b"x")
        (src / "train" / "labels" / f"im{i}.txt").write_text(
            "0 0.5 0.5 0.2 0.1\n1 0.3 0.3 0.1 0.2\n3 0.1 0.1 0.05 0.05\n2 0.9 0.2 0.05 0.2\n")
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, "-B", "training/remap_labels.py", "--src", str(src),
                        "--map", "training/label_map.example.yaml", "--out", str(out), "--split", "0.6", "0.2", "0.2"],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    data = yaml.safe_load(open(out / "data.yaml"))
    assert list(data["names"].values()) == ["swimmer_horizontal", "swimmer_vertical", "head_only", "underwater",
                                            "out_of_water"]
    assert {"train", "val", "test"} <= set(data)
    labels = list((out / "train" / "labels").glob("*.txt")) + list((out / "val" / "labels").glob("*.txt"))
    lines = [ln.split()[0] for f in labels for ln in f.read_text().splitlines()]
    assert set(lines) == {"0", "1", "4"}                  # swimming, drowning->vertical, person->deck; ball dropped
    assert len(list((out / "test" / "images").glob("*.jpg"))) == 2
