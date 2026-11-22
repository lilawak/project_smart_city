"""
Part 1 (Tasks 1.2 + 1.3): grab a webcam frame every N seconds, count people
with YOLOv8 and log "timestamp,people" to a CSV file.

BEST  --source url     read the stream directly: no screen involved, full resolution.
                       --url accepts:
                         - a direct stream link (.m3u8 / MJPEG / RTSP)
                         - a YouTube live link (needs: pip install yt-dlp)
                         - a local image or video file (handy for testing)
FALLBACK --source screen   drag a rectangle around the webcam on your screen.
                       The area is saved in region.json, so you select it once.

Examples:
  python capture_count.py --source url --url "https://.../playlist.m3u8"
  python capture_count.py --source url --url "https://www.youtube.com/watch?v=XXXX"
  python capture_count.py --source url --url ../material/image.png --duration 0.05
  python capture_count.py --source screen
  python capture_count.py --source screen --reselect     # choose the area again
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

PERSON_CLASS_ID = 0  # COCO class 0 = person


# ---------------------------------------------------------------- URL source
def resolve_url(url):
    """YouTube links are converted to a direct stream URL with yt-dlp."""
    if "youtube.com" in url or "youtu.be" in url:
        try:
            out = subprocess.run(
                [sys.executable, "-m", "yt_dlp", "-g", "-f", "best[height<=1080]/best", url],
                capture_output=True, text=True, check=True, timeout=90,
            )
            direct = out.stdout.strip().splitlines()[0]
            print("[INFO] YouTube link resolved to a direct stream.")
            return direct
        except Exception as e:
            raise RuntimeError(f"Could not resolve the YouTube link (did you run 'pip install yt-dlp'?): {e}")
    return url


def make_url_grabber(url):
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


# ------------------------------------------------------------- screen source
def load_or_select_region(region_file, monitor_index, delay, reselect):
    path = Path(region_file)
    if path.exists() and not reselect:
        print(f"[INFO] Using the saved area from {path} (add --reselect to choose again)")
        return json.loads(path.read_text())

    from mss import mss

    print(f"Bring the WEBCAM window to the front now. Taking a screenshot in {delay} seconds...")
    for s in range(delay, 0, -1):
        print(f"  {s}", flush=True)
        time.sleep(1)

    with mss() as sct:
        mon = sct.monitors[monitor_index]
        img = cv2.cvtColor(np.array(sct.grab(mon)), cv2.COLOR_BGRA2BGR)

    # Show the screenshot scaled down to fit; map the selection back to real pixels
    h, w = img.shape[:2]
    scale = min(1.0, 1400 / w, 800 / h)
    shown = cv2.resize(img, None, fx=scale, fy=scale) if scale < 1 else img
    print("DRAG a rectangle around the webcam image, then press ENTER (press C to cancel).")
    x, y, rw, rh = cv2.selectROI("Select the webcam area", shown, showCrosshair=False)
    cv2.destroyAllWindows()
    if rw == 0 or rh == 0:
        raise SystemExit("No area selected. Run again.")

    region = {
        "left": mon["left"] + int(x / scale),
        "top": mon["top"] + int(y / scale),
        "width": int(rw / scale),
        "height": int(rh / scale),
    }
    path.write_text(json.dumps(region))
    print(f"[INFO] Area saved to {path}: {region}")
    return region


def make_screen_grabber(region):
    from mss import mss

    sct = mss()

    def grab():
        # mss returns BGRA; YOLO/OpenCV want 3-channel BGR
        return cv2.cvtColor(np.array(sct.grab(region)), cv2.COLOR_BGRA2BGR)

    return grab


# ---------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["screen", "url"], default="url")
    p.add_argument("--url", help="stream link, YouTube link, or image/video file (with --source url)")
    p.add_argument("--referer", help="website address to send as Referer, if the stream refuses to open")
    p.add_argument("--interval", type=float, default=5.0, help="seconds between frames (default 5)")
    p.add_argument("--duration", type=float, default=0, help="minutes to run; 0 = until Ctrl+C")
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--conf", type=float, default=0.5, help="confidence threshold (default 0.5)")
    p.add_argument("--out", default="results", help="output folder (default ./results)")
    p.add_argument("--reselect", action="store_true", help="(screen) choose the area again")
    p.add_argument("--region-file", default="region.json", help="(screen) where the area is saved")
    p.add_argument("--monitor", type=int, default=1, help="(screen) monitor number, 1 = main screen")
    p.add_argument("--delay", type=int, default=5, help="(screen) seconds before the screenshot")
    args = p.parse_args()

    if args.source == "url" and not args.url:
        p.error("--url is required with --source url")

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
        if args.source == "url":
            grab = make_url_grabber(args.url)
        else:
            region = load_or_select_region(args.region_file, args.monitor, args.delay, args.reselect)
            grab = make_screen_grabber(region)
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
                        print(f"[INFO] Frame size: {fw}x{fh} (check results/first_frame.jpg)")
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
