"""AprilTag detection test bench for the XIAO ESP32-S3 Sense.

Runs the drone's OWN AprilTag detector (CDE1302_UAV_Group_3/laptop/apriltag_host.py,
which compiles the drone's C code) on raw 320x240 grayscale frames from the
XIAO, so what turns green here is what the drone would accept.

    python tag_test.py                     # COM11, repo on the Desktop
    python tag_test.py --port COM5 --repo "C:/path/to/CDE1302_UAV_Group_3"

The drone repo is only read. The compiled detector is cached in
esp_camera/.apriltag-host/.

Green = passes the drone's gate (hamming <= 1, margin > cutoff).
Red   = found but rejected (shows the margin, so you can see how close it was).

Keys
  e        auto exposure on/off            [ ]   exposure down / up (manual)
  g        auto gain on/off                - =   gain down / up (manual)
  b        brightness cycle (-2..2)        a     AE level cycle (-2..2, auto only)
  , .      margin cutoff down / up         d     quad_decimate cycle
  r        back to start settings          c     clear stats
           (start = drone settings + fixed exposure --exposure, default 150; -1 = auto)
  s        save the raw frame (PNG)        q/Esc quit
  v        view: fixed threshold / Otsu / adaptive / gray   j k   threshold down / up
  i        detect on the raw gray frame (default) or on the shown view
"""
import argparse
import collections
import struct
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import serial

HERE = Path(__file__).resolve().parent
DEFAULT_REPO = HERE.parents[1]   # this repo (laptop/apriltag_host.py)

DECIMATE_STEPS = [1.0, 1.5, 2.0, 3.0]
GREEN, RED, WHITE, BLACK, YELLOW = (0, 220, 0), (0, 0, 255), (255, 255, 255), (0, 0, 0), (0, 220, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def binarize(gray, tile=4, min_diff=5):
    """Black/white view of the frame, made the way the AprilTag detector does it.

    Display only: detection always runs on the gray frame. Each 4x4 tile gets
    its own threshold, halfway between the darkest and brightest pixel in the
    3x3 tiles around it. Where those are closer than min_diff the area is too
    flat to call, and it is shown mid-gray (the detector ignores it too).
    """
    h, w = gray.shape
    th, tw = h // tile, w // tile
    g = gray[:th * tile, :tw * tile]
    blocks = g.reshape(th, tile, tw, tile)
    k = np.ones((3, 3), np.uint8)
    lo = cv2.erode(np.ascontiguousarray(blocks.min(axis=(1, 3))), k)
    hi = cv2.dilate(np.ascontiguousarray(blocks.max(axis=(1, 3))), k)
    size = (tw * tile, th * tile)
    lo = cv2.resize(lo, size, interpolation=cv2.INTER_NEAREST).astype(np.int16)
    hi = cv2.resize(hi, size, interpolation=cv2.INTER_NEAREST).astype(np.int16)
    out = np.full_like(gray, 127)
    sub = np.where(g > lo + (hi - lo) // 2, 255, 0).astype(np.uint8)
    sub[(hi - lo) < min_diff] = 127
    out[:th * tile, :tw * tile] = sub
    return out


VIEW_MODES = ("fixed", "otsu", "adaptive", "gray")
DEFAULT_EXPOSURE = 150      # sensor units 0-1200 (shorter = less motion blur, darker); -1 = auto


def set_camera(ser, exposure):
    """Drone sensor settings, then a fixed exposure unless exposure < 0 (auto)."""
    cmds = ["drone"]
    if exposure >= 0:
        cmds += ["aec 0", f"exp {min(1200, exposure)}"]
    for c in cmds:
        ser.write((c + "\n").encode())
        time.sleep(0.05)


def make_view(gray, mode, thresh):
    """The picture shown on screen; detection never uses it.

    fixed     cv2.THRESH_BINARY at `thresh`: brighter -> white, else black
    otsu      THRESH_BINARY with the threshold picked per frame by Otsu's method
    adaptive  the detector's own per-tile threshold (binarize)
    gray      the raw frame
    Returns (image, label for the screen).
    """
    if mode == "fixed":
        return cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)[1], f"binary, threshold {thresh}"
    if mode == "otsu":
        t, img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return img, f"binary, Otsu picked {t:.0f}"
    if mode == "adaptive":
        return binarize(gray), "adaptive (detector's own method)"
    return gray, "gray"


def load_apriltag_host(repo: Path):
    laptop = repo / "laptop"
    if not (laptop / "apriltag_host.py").is_file():
        sys.exit(f"can't find {laptop / 'apriltag_host.py'}; pass --repo")
    sys.path.insert(0, str(laptop))
    import apriltag_host as ah
    ah.BUILD_DIR = HERE / ".apriltag-host"   # build here, never inside the drone repo
    return ah


class Reader(threading.Thread):
    """Reads the serial port; keeps only the newest frame so latency stays low."""

    def __init__(self, ser):
        super().__init__(daemon=True)
        self.ser = ser
        self.lock = threading.Lock()
        self.frame = None          # (seq, gray ndarray)
        self.seq = 0
        self.status = {}
        self.messages = collections.deque(maxlen=20)
        self.dropped = 0
        self.running = True

    def run(self):
        buf = bytearray()
        while self.running:
            try:
                buf += self.ser.read(self.ser.in_waiting or 1)
            except serial.SerialException as exc:
                self.messages.append(f"serial error: {exc}")
                return
            while True:
                i = buf.find(b"FRM!")
                if i < 0:
                    self._text(bytes(buf[:-3]))
                    del buf[:-3]
                    break
                if i > 0:
                    self._text(bytes(buf[:i]))
                    del buf[:i]
                if len(buf) < 12:
                    break
                w, h, n = struct.unpack("<HHI", buf[4:12])
                if n != w * h or n == 0 or n > 2_000_000:
                    del buf[:4]           # not a real header; resync
                    continue
                if len(buf) < 12 + n + 4:
                    break
                if buf[12 + n:16 + n] != b"END!":
                    self.dropped += 1     # bytes were lost; skip this frame
                    del buf[:4]
                    continue
                gray = np.frombuffer(bytes(buf[12:12 + n]), np.uint8).reshape(h, w)
                del buf[:16 + n]
                with self.lock:
                    self.seq += 1
                    self.frame = (self.seq, gray)

    def _text(self, data: bytes):
        for line in data.decode("ascii", errors="ignore").splitlines():
            line = line.strip()
            if line.startswith("STATUS "):
                kv = dict(p.split("=", 1) for p in line[7:].split() if "=" in p)
                with self.lock:
                    self.status = kv
            elif line.startswith("MSG "):
                self.messages.append(line[4:])
                print("[esp]", line[4:])

    def latest(self):
        with self.lock:
            return self.frame, dict(self.status)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="COM11")
    ap.add_argument("--repo", type=Path, default=DEFAULT_REPO, help="CDE1302_UAV_Group_3 folder")
    ap.add_argument("--scale", type=float, default=2.5, help="window size multiplier")
    ap.add_argument("--threshold", type=int, default=200, help="fixed binary threshold for the view, 0-255")
    ap.add_argument("--exposure", type=int, default=DEFAULT_EXPOSURE,
                    help="fixed exposure 0-1200 (shorter = less blur, darker); -1 = auto")
    args = ap.parse_args()

    ah = load_apriltag_host(args.repo)
    fw = ah.FIRMWARE_PARAMS
    decimate = fw.quad_decimate
    cutoff = ah.GATE_MIN_MARGIN

    def make_detector():
        params = ah.DetectorParams(quad_decimate=decimate, quad_sigma=fw.quad_sigma,
                                   refine_edges=fw.refine_edges,
                                   decode_sharpening=fw.decode_sharpening, nthreads=fw.nthreads)
        return ah.Detector(params=params, min_margin=cutoff)

    detector = make_detector()
    print(f"detector built: {detector.params.summary()}, {detector.ncodes} tag16h5 codes")

    ser = serial.Serial(args.port, 115200, timeout=0.2)
    reader = Reader(ser)
    reader.start()

    def send(cmd):
        ser.write((cmd + "\n").encode())

    time.sleep(0.3)
    set_camera(ser, args.exposure)   # drone sensor settings + fixed exposure

    # Rolling stats over the last few seconds
    hist = collections.deque()        # (time, accepted?, best margin or None)
    proc_hist = collections.deque(maxlen=30)
    last_seq, view = 0, None
    view_mode, thresh = 0, args.threshold    # v cycles VIEW_MODES
    detect_on_view = False                   # i: detect on the gray frame (default) or the shown view
    t_fps, n_fps, fps = time.time(), 0, 0.0
    win = "AprilTag test (drone detector)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print(__doc__.split("Keys", 1)[1])

    try:
        while True:
            frame, st = reader.latest()
            if frame and frame[0] != last_seq:
                last_seq, gray = frame
                n_fps += 1
                if time.time() - t_fps >= 1:
                    fps, n_fps, t_fps = n_fps / (time.time() - t_fps), 0, time.time()

                shown, view_label = make_view(gray, VIEW_MODES[view_mode], thresh)
                t0 = time.perf_counter()
                dets = detector.detect(shown if detect_on_view else gray)   # i toggles
                proc_hist.append((time.perf_counter() - t0) * 1000)

                h, w = gray.shape
                view = cv2.resize(cv2.cvtColor(shown, cv2.COLOR_GRAY2BGR),
                                  (round(w * args.scale), round(h * args.scale)),
                                  interpolation=cv2.INTER_NEAREST)
                accepted, best = False, None
                for d in dets:
                    ok = ah.passes_gate(d, cutoff)
                    if not ok and (d.hamming > 1 or d.decision_margin < 10):
                        continue      # tag16h5 noise; don't clutter the view
                    accepted |= ok
                    best = d.decision_margin if best is None else max(best, d.decision_margin)
                    pts = np.array([(round(x * args.scale - 0.5), round(y * args.scale - 0.5))
                                    for x, y in d.corners], np.int32)
                    col = GREEN if ok else RED
                    cv2.polylines(view, [pts], True, col, 2, cv2.LINE_AA)
                    label = f"id{d.id} m{d.decision_margin:.0f}"
                    if ok and d.range_m is not None:
                        label += f" {d.range_m:.2f}m"
                    org = (int(pts[:, 0].min()), max(15, int(pts[:, 1].min()) - 6))
                    cv2.putText(view, label, org, FONT, 0.55, BLACK, 3, cv2.LINE_AA)
                    cv2.putText(view, label, org, FONT, 0.55, col, 1, cv2.LINE_AA)

                now = time.time()
                hist.append((now, accepted, best))
                while hist and now - hist[0][0] > 3.0:
                    hist.popleft()

                # HUD
                rate = 100 * sum(a for _, a, _ in hist) / max(1, len(hist))
                margins = [m for _, _, m in hist if m is not None]
                exp_txt = "auto" if st.get("aec") == "1" else st.get("exp", "?")
                gain_txt = "auto" if st.get("agc") == "1" else st.get("gain", "?")
                lines = [
                    f"exposure {exp_txt}  gain {gain_txt}  bright {st.get('bright', '?')}  ae {st.get('ae', '?')}",
                    f"cutoff >{cutoff:.0f}  decimate {decimate:g}  detect {np.mean(proc_hist):.0f} ms (laptop)",
                    f"last 3s: tag accepted in {rate:.0f}% of frames"
                    + (f", best margin avg {np.mean(margins):.0f}" if margins else ""),
                    f"{fps:.1f} fps  dropped {reader.dropped}  sensor {st.get('pid', '?')}",
                    f"view: {view_label}   (v view, j/k threshold)",
                    f"detecting on: {'THIS VIEW' if detect_on_view else 'gray frame'}   (i toggle)",
                ]
                for k, txt in enumerate(lines):
                    y = 20 + 20 * k
                    cv2.putText(view, txt, (8, y), FONT, 0.5, BLACK, 3, cv2.LINE_AA)
                    cv2.putText(view, txt, (8, y), FONT, 0.5, YELLOW if k == 2 else WHITE, 1, cv2.LINE_AA)
                cv2.imshow(win, view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            exp, gain = int(st.get("exp", 300)), int(st.get("gain", 0))
            if key == ord("e"):
                send(f"aec {0 if st.get('aec') == '1' else 1}")
            elif key == ord("["):
                send(f"exp {max(0, int(exp / 1.4))}")
            elif key == ord("]"):
                send(f"exp {min(1200, max(exp + 1, int(exp * 1.4)))}")
            elif key == ord("g"):
                send(f"agc {0 if st.get('agc') == '1' else 1}")
            elif key == ord("-"):
                send(f"gain {max(0, gain - 2)}")
            elif key == ord("="):
                send(f"gain {min(30, gain + 2)}")
            elif key == ord("b"):
                send(f"bright {(int(st.get('bright', 1)) + 3) % 5 - 2}")
            elif key == ord("a"):
                send(f"ae {(int(st.get('ae', 0)) + 3) % 5 - 2}")
            elif key == ord(","):
                cutoff = max(0.0, cutoff - 5)
            elif key == ord("."):
                cutoff += 5
            elif key == ord("d"):
                decimate = DECIMATE_STEPS[(DECIMATE_STEPS.index(decimate) + 1) % len(DECIMATE_STEPS)] \
                    if decimate in DECIMATE_STEPS else 1.5
                detector.close()
                detector = make_detector()
                proc_hist.clear()
            elif key == ord("r"):
                set_camera(ser, args.exposure)
                cutoff, decimate = ah.GATE_MIN_MARGIN, fw.quad_decimate
                detector.close()
                detector = make_detector()
            elif key == ord("c"):
                hist.clear()
            elif key == ord("v"):
                view_mode = (view_mode + 1) % len(VIEW_MODES)
            elif key == ord("i"):
                detect_on_view = not detect_on_view
            elif key == ord("j"):
                thresh = max(0, thresh - 5)
            elif key == ord("k"):
                thresh = min(255, thresh + 5)
            elif key == ord("s") and frame:
                name = HERE / time.strftime("frame_%Y%m%d_%H%M%S.png")
                cv2.imwrite(str(name), frame[1])
                print("saved", name)
            if key != 255:
                hist.clear() if key in map(ord, "eg[]-=ba,.dr") else None
    except KeyboardInterrupt:
        pass
    finally:
        reader.running = False
        print(f"\nfinal settings: exposure {st.get('exp')} (aec={st.get('aec')}), gain {st.get('gain')} "
              f"(agc={st.get('agc')}), bright {st.get('bright')}, ae {st.get('ae')}, "
              f"cutoff {cutoff:g}, decimate {decimate:g}")
        detector.close()
        ser.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
