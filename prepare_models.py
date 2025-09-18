from pathlib import Path
import requests

PROTO_URL = "https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/hand/pose_deploy.prototxt"
CAFFE_URLS = [
    # Primary mirror (HuggingFace)
    "https://huggingface.co/camenduru/openpose/resolve/main/pose_iter_102000.caffemodel?download=true",
    # Fallback (Dropbox)
    "https://www.dropbox.com/s/gqgsme6sgoo0zxf/pose_iter_102000.caffemodel?dl=1",
]

def download(url, dst: Path, min_mb=None, timeout=60):
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"→ Downloading: {url}")
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
    size_mb = dst.stat().st_size / (1024 * 1024)
    print(f"  saved: {dst} ({size_mb:.1f} MB)")
    if (min_mb is not None) and (size_mb < min_mb * 0.9):
        raise RuntimeError(f"File too small ({size_mb:.1f} MB) vs expected {min_mb} MB")

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Download OpenPose hand model files.")
    ap.add_argument("--models", type=Path, default=Path("models/hand"), help="Model directory")
    args = ap.parse_args()

    proto_path = args.models / "pose_deploy.prototxt"
    caffe_path = args.models / "pose_iter_102000.caffemodel"

    if not proto_path.exists():
        download(PROTO_URL, proto_path)
    else:
        print(f"✓ Found: {proto_path}")

    if (not caffe_path.exists()) or (caffe_path.stat().st_size < 120 * 1024 * 1024):
        last_error = None
        for url in CAFFE_URLS:
            try:
                download(url, caffe_path, min_mb=147)
                last_error = None
                break
            except Exception as e:
                print(f"  failed from this source: {e}")
                last_error = e
        if last_error:
            raise last_error
    else:
        print(f"✓ Found: {caffe_path} ({caffe_path.stat().st_size/1024/1024:.1f} MB)")

if __name__ == "__main__":
    main()
