"""Download the public test clips into samples/ (and make the synthetic ones).

    python get_samples.py            # Intel sample videos (CC BY 4.0) + synthetic test clips
    python get_samples.py --no-intel # only the synthetic clips (no internet needed)

Intel IoT DevKit sample videos: https://github.com/intel-iot-devkit/sample-videos (CC BY 4.0).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "samples"
BASE = "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/"
INTEL = {
    "one-by-one-person-detection.mp4": "people stop at a table one by one (loitering + zone demo), 3.3 MB",
    "person-bicycle-car-detection.mp4": "car park with people, bikes, cars (traffic preset), 6.0 MB",
    "worker-zone-detection.mp4": "warehouse worker, 1080p 59.94 fps (odd frame rate test), 12.6 MB",
}


def download(name: str) -> None:
    """Fetch one clip unless it is already there."""
    dst = SAMPLES / name
    if dst.exists() and dst.stat().st_size > 0:
        print(f"  have {name}")
        return
    print(f"  downloading {name} ...", flush=True)
    tmp = dst.with_suffix(".part")
    urllib.request.urlretrieve(BASE + name, tmp)
    tmp.replace(dst)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Get the test clips into samples/")
    ap.add_argument("--no-intel", action="store_true", help="skip the Intel downloads")
    args = ap.parse_args(argv)
    SAMPLES.mkdir(exist_ok=True)
    if not args.no_intel:
        print("Intel sample videos (CC BY 4.0):")
        for name, what in INTEL.items():
            print(f"  - {name}: {what}")
            download(name)
    print("Synthetic test clips (drawn locally, with ground-truth labels):")
    py = sys.executable
    subprocess.run([py, str(ROOT / "tests" / "synthetic.py"), "--out-video", str(SAMPLES / "synthetic.mp4"),
                    "--out-dir", str(ROOT / "outputs" / "synthetic")], check=True)
    subprocess.run([py, str(ROOT / "tests" / "synthetic_safety.py")], check=True)
    print("Done. Try:  python run.py --video samples/synthetic_safety.mp4 --scenario workplace "
          "--out outputs/synthetic_safety --reuse")
    return 0


if __name__ == "__main__":
    sys.exit(main())
