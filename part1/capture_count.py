"""
Part 1 (Tasks 1.2 + 1.3): grab a webcam frame every N seconds, count people
with YOLOv8 and log "timestamp,people" to a CSV file.

--url accepts:
  - a direct stream link (.m3u8 / MJPEG / RTSP)
  - a YouTube live link (needs: pip install yt-dlp)
  - a local image or video file (handy for testing)

Examples:
  python capture_count.py --url "https://.../playlist.m3u8"
  python capture_count.py --url "https://www.youtube.com/watch?v=XXXX"
  python capture_count.py --url ../material/image.png --duration 0.05
"""
import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

PERSON_CLASS_ID = 0  # COCO class 0 = person


def resolve_url(url):
    """YouTube links are converted to a direct stream URL with yt-dlp."""
    if "youtube.com" in url or "youtu.be" in url:
        try:
            out = subprocess.run(
                [sys.executable, "-m", "yt_dlp", "-g", "-f", "bv*[height<=1080]/bv*/b", url],
                capture_output=True, text=True, check=True, timeout=90,
            )
            direct = out.stdout.strip().splitlines()[0]
            print("[INFO] YouTube link resolved to a direct stream.")
            return direct
        except Exception as e:
            raise RuntimeError(f"Could not resolve the YouTube link (did you run 'pip install yt-dlp'?): {e}")
    return url


def make_grabber(url):
    state = {"url": resolve_url(url), "fails": 0}

    def grab():
        # Re-open each time so we get the CURRENT frame, not a stale buffered one
        cap = cv2.VideoCapture(state["url"])
        try:
            ok, frame = cap.read()
        finally:
            cap.release()
        if ok:
            state["fails"] = 0
            return frame
        state["fails"] += 1
        if state["fails"] >= 3:  # YouTube stream links expire after a few hours
            state["fails"] = 0
            try:
                state["url"] = resolve_url(url)
            except RuntimeError as e:
                print(f"[WARN] {e}")
        return None

    return grab


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", required=True, help="stream link, YouTube link, or image/video file")
    p.add_argument("--referer", help="website address to send as Referer, if the stream refuses to open")
    p.add_argument("--interval", type=float, default=5.0, help="seconds between frames (default 5)")
    p.add_argument("--duration", type=float, default=0, help="minutes to run; 0 = until Ctrl+C")
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--conf", type=float, default=0.5, help="confidence threshold (default 0.5)")
    p.add_argument("--out", default="results", help="output folder (default ./results)")
    args = p.parse_args()

    if args.referer:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            f"headers;Referer: {args.referer}\r\nUser-Agent: Mozilla/5.0\r\n"
        )

    out = Path(args.out)
    raw_dir = out / "frames_raw"
    ann_dir = out / "frames_annotated"
    raw_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out / "people_count.csv"
    new_file = not csv_path.exists()

    try:
        grab = make_grabber(args.url)
    except RuntimeError as e:
        p.error(str(e))

    model = YOLO(args.model)

    end = time.monotonic() + args.duration * 60 if args.duration else None
    next_t = time.monotonic()
    first = True
    print(f"Capturing every {args.interval}s -> {csv_path}  (Ctrl+C to stop)")

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["timestamp", "people"])
        try:
            while end is None or time.monotonic() < end:
                frame = grab()
                now = datetime.now()
                if frame is None:
                    print(f"[{now:%H:%M:%S}] WARN: could not read a frame, retrying next cycle")
                else:
                    if first:
                        fh, fw = frame.shape[:2]
                        cv2.imwrite(str(out / "first_frame.jpg"), frame)
                        print(f"[INFO] Frame size: {fw}x{fh} (check {out / 'first_frame.jpg'})")
                        first = False
                    res = model.predict(frame, classes=[PERSON_CLASS_ID], conf=args.conf, verbose=False)[0]
                    n = len(res.boxes)
                    writer.writerow([now.strftime("%Y-%m-%d %H:%M:%S"), n])
                    f.flush()
                    stamp = f"{now:%Y%m%d_%H%M%S}"
                    cv2.imwrite(str(raw_dir / f"{stamp}.jpg"), frame)
                    cv2.imwrite(str(ann_dir / f"{stamp}_{n}p.jpg"), res.plot())
                    print(f"[{now:%H:%M:%S}] people: {n}")
                next_t += args.interval
                time.sleep(max(0.0, next_t - time.monotonic()))
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()