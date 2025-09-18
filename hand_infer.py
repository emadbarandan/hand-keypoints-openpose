import os, glob, csv
from pathlib import Path
import numpy as np
import cv2

N_POINTS = 22
POSE_PAIRS = [
    [0,1],[1,2],[2,3],[3,4],
    [0,5],[5,6],[6,7],[7,8],
    [0,9],[9,10],[10,11],[11,12],
    [0,13],[13,14],[14,15],[15,16],
    [0,17],[17,18],[18,19],[19,20]
]
FINGERS = [
    [1,2,3,4],    # thumb
    [5,6,7,8],    # index
    [9,10,11,12], # middle
    [13,14,15,16],# ring
    [17,18,19,20] # little
]

def load_net(protoFile, weightsFile):
    assert Path(protoFile).exists(),  f"Not found: {protoFile}"
    assert Path(weightsFile).exists(),f"Not found: {weightsFile}"
    net = cv2.dnn.readNetFromCaffe(str(protoFile), str(weightsFile))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net

def run_network(net, frame, inH=368, pt_thresh=0.20):
    H, W = frame.shape[:2]
    inW = int((W / H * inH) // 8 * 8)
    blob = cv2.dnn.blobFromImage(frame, 1.0/255, (inW, inH), (0,0,0), swapRB=False, crop=False)
    net.setInput(blob)
    out = net.forward()

    probs, points = [], []
    for i in range(N_POINTS):
        pm = out[0, i, :, :]
        pm = cv2.resize(pm, (W, H))
        _, p, _, pt = cv2.minMaxLoc(pm)
        probs.append(float(p))
        if p > pt_thresh:
            points.append((int(pt[0]), int(pt[1])))
        else:
            points.append(None)
    return points, probs

def bbox_from_points(points):
    xs, ys = [], []
    for p in points:
        if p is not None:
            xs.append(p[0]); ys.append(p[1])
    if not xs:
        return None, 0
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    w, h = x2 - x1, y2 - y1
    return (x1, y1, w, h), w*h

def skeleton_valid_edges(points):
    cnt = 0
    for a, b in POSE_PAIRS:
        if (points[a] is not None) and (points[b] is not None):
            cnt += 1
    return cnt

def finger_has_chain(points):
    for chain in FINGERS:
        present = [points[i] is not None for i in chain]
        for s in range(len(chain) - 2):
            if present[s] and present[s + 1] and present[s + 2]:
                return True
    return False

def is_hand_present(points, probs, img_area, mean_prob_thresh=0.22):
    detected_points = [p for p in points if p is not None]
    if len(detected_points) < 8:
        return False
    mean_prob = np.mean([probs[i] for i, p in enumerate(points) if p is not None]) if detected_points else 0.0
    if mean_prob < mean_prob_thresh:
        return False
    if points[0] is None:
        return False  # wrist must be present (id=0)
    if skeleton_valid_edges(points) < 4:
        return False
    bbox, area = bbox_from_points(points)
    if bbox is None:
        return False
    frac = area / float(img_area)
    if not (0.005 <= frac <= 0.5):  # between 0.5% and 50% of image
        return False
    if not finger_has_chain(points):
        return False
    return True

def draw_outputs(frame, points):
    frameKP = frame.copy()
    for i, pt in enumerate(points):
        if pt is not None:
            cv2.circle(frameKP, pt, 8, (0,255,255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frameKP, str(i), pt, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, lineType=cv2.LINE_AA)

    frameSK = frame.copy()
    for a, b in POSE_PAIRS:
        if (points[a] is not None) and (points[b] is not None):
            cv2.line(frameSK, points[a], points[b], (0,255,255), 2)
            cv2.circle(frameSK, points[a], 8, (0,0,255), thickness=-1, lineType=cv2.FILLED)
            cv2.circle(frameSK, points[b], 8, (0,0,255), thickness=-1, lineType=cv2.FILLED)
    return frameKP, frameSK

def infer_handedness(points):
    thumb = points[1]
    index = points[5]
    if (thumb is None) or (index is None):
        return None
    # Note: mirrored images may invert left/right.
    return "right" if thumb[0] < index[0] else "left"

def main():
    import argparse
    ap = argparse.ArgumentParser(description="CPU-only OpenPose hand inference via OpenCV DNN.")
    ap.add_argument("--images", type=Path, default=Path("data/5"), help="Folder with images (subfolders supported)")
    ap.add_argument("--output", type=Path, default=Path("outputs"), help="Output folder")
    ap.add_argument("--models", type=Path, default=Path("models/hand"), help="Folder with prototxt & caffemodel")
    ap.add_argument("--inH", type=int, default=368, help="Network input height, e.g., 256 for faster CPU runs")
    ap.add_argument("--pt-thresh", type=float, default=0.20, help="Per-keypoint detection threshold")
    ap.add_argument("--mean-thresh", type=float, default=0.22, help="Mean confidence threshold")
    args = ap.parse_args()

    protoFile   = args.models / "pose_deploy.prototxt"
    weightsFile = args.models / "pose_iter_102000.caffemodel"
    net = load_net(protoFile, weightsFile)

    root_img_dir = args.images
    root_out_dir = args.output
    root_out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = root_out_dir / "hand_detection_report.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["p", "frame", "sx_model", "dx_model"])

        subfolders = [p for p in sorted(root_img_dir.glob("*")) if p.is_dir()]
        if not subfolders:
            subfolders = [root_img_dir]

        for p_dir in subfolders:
            p_name = p_dir.name
            out_dir = root_out_dir / p_name
            out_dir.mkdir(parents=True, exist_ok=True)

            img_list = []
            for ext in ("*.jpg","*.png","*.jpeg","*.bmp","*.JPG","*.PNG","*.JPEG","*.BMP"):
                img_list += list(p_dir.glob(ext))
            img_list = sorted(img_list)

            print(f"[p={p_name}] Found {len(img_list)} images")
            for img_path in img_list:
                frame = cv2.imread(str(img_path))
                if frame is None:
                    print("Skipping unreadable:", img_path)
                    continue

                H, W = frame.shape[:2]
                points, probs = run_network(net, frame, inH=args.inH, pt_thresh=args.pt_thresh)
                present = is_hand_present(points, probs, H*W, mean_prob_thresh=args.mean_thresh)

                frameKP, frameSK = draw_outputs(frame, points)
                base = img_path.stem
                cv2.imwrite(str(out_dir / f"{base}_keypoints.jpg"), frameKP)
                cv2.imwrite(str(out_dir / f"{base}_skeleton.jpg"), frameSK)

                if present:
                    side = infer_handedness(points)
                    if side == "left":  sx, dx = 0, 1
                    elif side == "right": sx, dx = 1, 0
                    else: sx, dx = 1, 1
                else:
                    sx, dx = 1, 1

                writer.writerow([p_name, img_path.name, sx, dx])
                print(f"  {base}: present={present}, sx={sx}, dx={dx}")

    print("✅ CSV:", csv_path)
    print("✅ Outputs:", root_out_dir)

if __name__ == "__main__":
    main()
