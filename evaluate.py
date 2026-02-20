"""
Full evaluation on validation/test set with ground-truth labels.
Computes per-class Precision, Recall, F1, mAP@50.

Usage (local):
    python evaluate.py --images dataset/dataset/valid/images \
                       --labels dataset/dataset/valid/labels \
                       --outdir  eval_output

Usage (Docker):
    docker run --rm --cpus=4 --memory=4g \
        -v D:/rpi_yolo_test:/app/project \
        yolo-rpi python /app/project/evaluate.py \
            --images /app/project/dataset/dataset/valid/images \
            --labels /app/project/dataset/dataset/valid/labels \
            --outdir  /app/project/eval_output
"""

import argparse
import csv
import os
import time

import cv2
import numpy as np
import onnxruntime as ort

# ── config ────────────────────────────────────────────────────────────────────
MODEL_PATH  = "best.onnx"
IMG_SIZE    = 640
CONF_THRESH = 0.25
IOU_THRESH  = 0.45
IOU_MATCH   = 0.50          # IoU threshold to count a detection as TP

CLASS_NAMES = ["cracks", "good_road", "open_manhole", "pothole"]
COLORS      = np.random.default_rng(42).integers(0, 255, (len(CLASS_NAMES), 3)).tolist()
IMG_EXTS    = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# ─────────────────────────────────────────────────────────────────────────────


def load_model(path):
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    print(f"[INFO] Model  : {path}")
    print(f"[INFO] Input  : {name} {sess.get_inputs()[0].shape}")
    print(f"[INFO] Output : {sess.get_outputs()[0].name} {sess.get_outputs()[0].shape}\n")
    return sess, name


def preprocess(frame, img_size):
    h, w = frame.shape[:2]
    scale = img_size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(frame, (nw, nh))
    pad_h, pad_w = img_size - nh, img_size - nw
    top, left = pad_h // 2, pad_w // 2
    padded = cv2.copyMakeBorder(resized, top, pad_h - top, left, pad_w - left,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    blob = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))[np.newaxis]
    return blob, scale, (left, top)


def postprocess(output, orig_shape, scale, pad, conf_thresh, iou_thresh):
    pred = output[0].T                      # (8400, 4+C)
    boxes_xywh   = pred[:, :4]
    class_scores = pred[:, 4:]
    class_ids    = np.argmax(class_scores, axis=1)
    confidences  = class_scores[np.arange(len(class_ids)), class_ids]

    mask = confidences >= conf_thresh
    boxes_xywh, confidences, class_ids = boxes_xywh[mask], confidences[mask], class_ids[mask]
    if len(boxes_xywh) == 0:
        return []

    px, py = pad
    x1 = (boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2 - px) / scale
    y1 = (boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2 - py) / scale
    x2 = (boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2 - px) / scale
    y2 = (boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2 - py) / scale
    oh, ow = orig_shape
    x1, y1 = np.clip(x1, 0, ow), np.clip(y1, 0, oh)
    x2, y2 = np.clip(x2, 0, ow), np.clip(y2, 0, oh)

    bboxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
    scores = confidences.tolist()
    indices = cv2.dnn.NMSBoxes(bboxes, scores, conf_thresh, iou_thresh)
    dets = []
    for i in (indices.flatten() if len(indices) else []):
        bx, by, bw, bh = bboxes[i]
        dets.append({
            "box":      (int(bx), int(by), int(bx + bw), int(by + bh)),
            "score":    float(scores[i]),
            "class_id": int(class_ids[i]),
        })
    return dets


def load_gt_labels(label_path, img_w, img_h):
    """Read YOLO-format .txt and convert to pixel boxes."""
    gts = []
    if not os.path.exists(label_path):
        return gts
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls, cx, cy, bw, bh = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = int((cx - bw / 2) * img_w)
            y1 = int((cy - bh / 2) * img_h)
            x2 = int((cx + bw / 2) * img_w)
            y2 = int((cy + bh / 2) * img_h)
            gts.append({"class_id": cls, "box": (x1, y1, x2, y2)})
    return gts


def box_iou(a, b):
    """IoU between two (x1,y1,x2,y2) boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


def match_detections(dets, gts, iou_thresh):
    """
    Match predictions to ground truths.
    Returns lists of (score, is_tp) sorted by descending score, and total gt count.
    """
    matched_gt = set()
    results = []   # (score, is_tp)
    for det in sorted(dets, key=lambda d: -d["score"]):
        best_iou, best_idx = 0, -1
        for gi, gt in enumerate(gts):
            if gi in matched_gt:
                continue
            if gt["class_id"] != det["class_id"]:
                continue
            iou = box_iou(det["box"], gt["box"])
            if iou > best_iou:
                best_iou, best_idx = iou, gi
        if best_iou >= iou_thresh:
            matched_gt.add(best_idx)
            results.append((det["score"], True))
        else:
            results.append((det["score"], False))
    return results, len(gts)


def compute_ap(scores_tp, n_gt):
    """Compute Average Precision from sorted (score, is_tp) list."""
    if n_gt == 0:
        return float("nan")
    scores_tp = sorted(scores_tp, key=lambda x: -x[0])
    tp_cum, fp_cum = 0, 0
    precisions, recalls = [], []
    for _, is_tp in scores_tp:
        if is_tp:
            tp_cum += 1
        else:
            fp_cum += 1
        precisions.append(tp_cum / (tp_cum + fp_cum))
        recalls.append(tp_cum / n_gt)
    # VOC 11-point interpolation
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        p = [precisions[i] for i in range(len(recalls)) if recalls[i] >= t]
        ap += max(p) if p else 0.0
    return ap / 11.0


def draw(frame, dets, gts):
    # Draw GT in blue
    for g in gts:
        x1, y1, x2, y2 = g["box"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 1)
        cv2.putText(frame, f"GT:{CLASS_NAMES[g['class_id']]}", (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 100, 0), 1)
    # Draw predictions in green/red
    for d in dets:
        x1, y1, x2, y2 = d["box"]
        color = COLORS[d["class_id"] % len(COLORS)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{CLASS_NAMES[d['class_id']]} {d['score']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y2 + 2), (x1 + tw, y2 + th + 6), color, -1)
        cv2.putText(frame, label, (x1, y2 + th + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def main():
    parser = argparse.ArgumentParser(description="YOLOv8n ONNX evaluation with mAP")
    parser.add_argument("--images",  required=True, help="Folder of test images")
    parser.add_argument("--labels",  default=None,  help="Folder of YOLO .txt labels (optional)")
    parser.add_argument("--outdir",  default="eval_output", help="Where to save results")
    parser.add_argument("--save-images", action="store_true", default=True,
                        help="Save annotated images (default: True)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Auto-detect labels folder if not given
    labels_dir = args.labels
    if labels_dir is None:
        candidate = args.images.replace("images", "labels")
        if os.path.isdir(candidate):
            labels_dir = candidate
            print(f"[INFO] Auto-detected labels: {labels_dir}")

    sess, input_name = load_model(MODEL_PATH)

    images = sorted([
        os.path.join(args.images, f) for f in os.listdir(args.images)
        if os.path.splitext(f)[1].lower() in IMG_EXTS
    ])
    print(f"[EVAL] {len(images)} images  |  labels: {'YES' if labels_dir else 'NO'}\n")

    # Per-class accumulator: {class_id: [(score, is_tp), ...]}
    class_scores = {i: [] for i in range(len(CLASS_NAMES))}
    class_n_gt   = {i: 0  for i in range(len(CLASS_NAMES))}

    csv_rows  = []
    latencies = []
    total_tp  = total_fp = total_gt = 0

    for idx, img_path in enumerate(images):
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        h, w = frame.shape[:2]

        blob, scale, pad = preprocess(frame, IMG_SIZE)
        t0 = time.perf_counter()
        output = sess.run(None, {input_name: blob})
        lat_ms = (time.perf_counter() - t0) * 1000
        latencies.append(lat_ms)

        dets = postprocess(output[0], (h, w), scale, pad, CONF_THRESH, IOU_THRESH)

        # Load GT labels
        gts = []
        if labels_dir:
            stem     = os.path.splitext(os.path.basename(img_path))[0]
            lbl_path = os.path.join(labels_dir, stem + ".txt")
            gts      = load_gt_labels(lbl_path, w, h)

        # Match and accumulate per class
        match_results, n_gt_img = match_detections(dets, gts, IOU_MATCH)
        img_tp = sum(1 for _, t in match_results if t)
        img_fp = sum(1 for _, t in match_results if not t)
        img_fn = n_gt_img - img_tp

        for det in dets:
            cid = det["class_id"]
            if cid < len(CLASS_NAMES):
                is_tp = any(
                    gt["class_id"] == cid and box_iou(det["box"], gt["box"]) >= IOU_MATCH
                    for gt in gts
                )
                class_scores[cid].append((det["score"], is_tp))
        for gt in gts:
            cid = gt["class_id"]
            if cid < len(CLASS_NAMES):
                class_n_gt[cid] += 1

        total_tp += img_tp
        total_fp += img_fp
        total_gt += n_gt_img

        det_str = ", ".join(
            f"{CLASS_NAMES[d['class_id']]}({d['score']:.2f})" for d in dets
        ) or "none"
        gt_str = ", ".join(
            f"{CLASS_NAMES[g['class_id']]}" for g in gts
            if g["class_id"] < len(CLASS_NAMES)
        ) or "none"

        print(f"  [{idx+1:4d}/{len(images)}] {os.path.basename(img_path):45s} "
              f"{lat_ms:6.1f}ms  pred:{det_str:40s}  gt:{gt_str}")

        csv_rows.append({
            "image":      os.path.basename(img_path),
            "latency_ms": round(lat_ms, 2),
            "predictions": det_str,
            "ground_truth": gt_str,
            "TP": img_tp, "FP": img_fp, "FN": img_fn,
        })

        # Save annotated image
        if args.save_images:
            annotated = draw(frame.copy(), dets, gts)
            cv2.putText(annotated,
                        f"{lat_ms:.1f}ms  TP:{img_tp} FP:{img_fp} FN:{img_fn}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            out_path = os.path.join(args.outdir, "out_" + os.path.basename(img_path))
            cv2.imwrite(out_path, annotated)

    # ── Per-class metrics ────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print(f"{'CLASS':<15} {'GT':>6} {'Preds':>6} {'TP':>5} {'AP@50':>8}")
    print("-" * 70)
    aps = []
    for cid, name in enumerate(CLASS_NAMES):
        n_gt  = class_n_gt[cid]
        preds = class_scores[cid]
        ap    = compute_ap(preds, n_gt)
        tp    = sum(1 for _, t in preds if t)
        aps.append(ap if not np.isnan(ap) else 0.0)
        ap_str = f"{ap:.4f}" if not np.isnan(ap) else "  N/A "
        print(f"  {name:<13} {n_gt:>6} {len(preds):>6} {tp:>5} {ap_str:>8}")

    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    rec  = total_tp / total_gt if total_gt > 0 else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    mean_ap = np.mean(aps)

    print("=" * 70)
    print(f"  {'mAP@50':<20} {mean_ap:.4f}")
    print(f"  {'Precision':<20} {prec:.4f}")
    print(f"  {'Recall':<20} {rec:.4f}")
    print(f"  {'F1 Score':<20} {f1:.4f}")
    print(f"  {'Avg latency':<20} {np.mean(latencies):.1f} ms")
    print(f"  {'Avg FPS':<20} {1000/np.mean(latencies):.2f}")
    print(f"  {'Images':<20} {len(latencies)}")
    print("=" * 70)

    # ── Save CSV ─────────────────────────────────────────────────────────────
    csv_path = os.path.join(args.outdir, "eval_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "latency_ms", "predictions",
                                          "ground_truth", "TP", "FP", "FN"])
        w.writeheader()
        w.writerows(csv_rows)

    # ── Save summary ─────────────────────────────────────────────────────────
    summary_path = os.path.join(args.outdir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("YOLOv8n ONNX Evaluation Summary\n")
        f.write("=" * 70 + "\n")
        for cid, name in enumerate(CLASS_NAMES):
            n_gt  = class_n_gt[cid]
            preds = class_scores[cid]
            ap    = compute_ap(preds, n_gt)
            tp    = sum(1 for _, t in preds if t)
            f.write(f"  {name:<13} GT:{n_gt:>5}  Preds:{len(preds):>5}  TP:{tp:>5}  AP@50:{ap:.4f}\n")
        f.write("=" * 70 + "\n")
        f.write(f"  mAP@50     : {mean_ap:.4f}\n")
        f.write(f"  Precision  : {prec:.4f}\n")
        f.write(f"  Recall     : {rec:.4f}\n")
        f.write(f"  F1 Score   : {f1:.4f}\n")
        f.write(f"  Avg latency: {np.mean(latencies):.1f} ms\n")
        f.write(f"  Avg FPS    : {1000/np.mean(latencies):.2f}\n")
        f.write(f"  Images     : {len(latencies)}\n")

    print(f"\n  CSV    → {csv_path}")
    print(f"  Report → {summary_path}")
    if args.save_images:
        print(f"  Images → {args.outdir}/out_*")


if __name__ == "__main__":
    main()
