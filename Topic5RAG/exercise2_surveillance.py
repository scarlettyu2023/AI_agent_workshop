import os
import cv2
import time
import json
import argparse
import requests
from tqdm import tqdm

OLLAMA_URL = "http://localhost:11434/api/generate"

def extract_frames(video_path: str, out_dir: str, every_seconds: float = 2.0):
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    interval = max(1, int(round(fps * every_seconds)))

    frames = []
    frame_idx = 0
    saved = 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    pbar = tqdm(total=total_frames if total_frames > 0 else None, desc="Extracting frames")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            t_sec = frame_idx / fps
            fname = os.path.join(out_dir, f"frame_{saved:05d}_{t_sec:.2f}s.jpg")
            cv2.imwrite(fname, frame)
            frames.append((fname, t_sec))
            saved += 1
        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    return frames

def ask_llava(image_path: str, question: str, model: str = "llava"):
    # Ollama API expects base64 image if using /api/generate? Actually Ollama accepts "images":[base64]
    # We'll encode to base64.
    import base64
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "model": model,
        "prompt": question,
        "images": [b64],
        "stream": False,
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return (data.get("response") or "").strip()

def normalize_yes_no(text: str):
    t = text.lower()
    # very forgiving parsing
    yes_markers = ["yes", "yeah", "yep", "a person", "there is a person", "someone", "human", "man", "woman"]
    no_markers = ["no", "nope", "none", "nobody", "no person", "not a person", "no one", "without a person"]

    # If explicitly starts with yes/no, trust it
    if t.startswith("yes"):
        return True
    if t.startswith("no"):
        return False

    # marker vote
    score = 0
    for m in yes_markers:
        if m in t:
            score += 1
    for m in no_markers:
        if m in t:
            score -= 1
    return score > 0

def find_intervals(detections):
    """
    detections: list of (t_sec, has_person_bool)
    Return intervals where has_person is True: [(start, end), ...]
    """
    intervals = []
    in_seg = False
    start = None

    for (t, has_person) in detections:
        if has_person and not in_seg:
            in_seg = True
            start = t
        elif (not has_person) and in_seg:
            in_seg = False
            end = t
            intervals.append((start, end))
            start = None

    # if video ends while still in segment
    if in_seg and start is not None:
        intervals.append((start, detections[-1][0]))
    return intervals

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to video file")
    ap.add_argument("--outdir", default="frames_out", help="Directory to store extracted frames")
    ap.add_argument("--every", type=float, default=2.0, help="Extract 1 frame every N seconds")
    ap.add_argument("--model", default="llava", help="Ollama model name (e.g., llava)")
    ap.add_argument("--question", default="Answer only YES or NO: Is there a person visible in this image?",
                    help="Prompt to ask VLM")
    ap.add_argument("--save_json", default="results.json", help="Where to save raw detections")
    args = ap.parse_args()

    # 1) Extract frames
    frames = extract_frames(args.video, args.outdir, every_seconds=args.every)
    print(f"Extracted {len(frames)} frames into {args.outdir}")

    # 2) Run VLM on each frame
    detections = []
    for (img_path, t_sec) in tqdm(frames, desc="Running LLaVA"):
        try:
            resp = ask_llava(img_path, args.question, model=args.model)
            has_person = normalize_yes_no(resp)
        except Exception as e:
            resp = f"ERROR: {e}"
            has_person = False
        detections.append({"time_sec": t_sec, "image": img_path, "has_person": has_person, "raw": resp})

    # 3) Save raw results
    with open(args.save_json, "w", encoding="utf-8") as f:
        json.dump(detections, f, indent=2, ensure_ascii=False)
    print(f"Saved raw detections to {args.save_json}")

    # 4) Compute intervals
    det_simple = [(d["time_sec"], d["has_person"]) for d in detections]
    intervals = find_intervals(det_simple)

    print("\n=== Person present intervals (approx) ===")
    if not intervals:
        print("No person detected.")
    else:
        for (s, e) in intervals:
            print(f"Person present from ~{s:.1f}s to ~{e:.1f}s")

if __name__ == "__main__":
    main()