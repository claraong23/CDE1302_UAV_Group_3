"""Checkerboard camera calibration for the XIAO (320x240 grayscale, same as the drone).

Finds the camera's real focal length (fx, fy), image centre (cx, cy) and lens
distortion, and writes them to calib_xiao.yaml. loc_test.py then uses them
instead of the drone's numbers + focal_scale.

1. Make and print a board (A4, print at 100% / "actual size", no "fit to page"):
       python cam_calib.py --make-board
   -> checkerboard_9x6_25mm.png. Tape it FLAT to something stiff (board, box).
   Measure one square with a ruler and pass it as --square (metres).
2. Calibrate:
       python cam_calib.py --square 0.025
   A live window shows the board's corners when found. A picture is taken
   automatically whenever the board is found in a NEW position (or press SPACE).
   Move the board, then HOLD IT STILL until the picture flashes (a moving board
   gives skewed, blurred pictures). Cover: near/far, all corners and edges of the
   picture, tilted BOTH ways left/right and up/down. The bottom line of the
   window lists what is still missing; it calibrates by itself once nothing is
   missing (and at least --shots pictures). Press c to calibrate early, q to quit.

Keys: SPACE take picture now   c or q calibrate   u undo last picture   Esc quit without calibrating

Every picture is saved to calib_xiao_shots.npz as it is taken; recalibrate from it any time:
    python cam_calib.py --square 0.024 --load calib_xiao_shots.npz
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import serial
import yaml

from tag_test import DEFAULT_EXPOSURE, DEFAULT_REPO, Reader, load_apriltag_host, set_camera

HERE = Path(__file__).resolve().parent
SCALE = 2


def make_board(cols, rows, square_mm, path):
    """Inner corners cols x rows -> (cols+1) x (rows+1) squares, A4 at 300 dpi."""
    dpi = 300
    px = int(round(square_mm / 25.4 * dpi))
    w, h = int(210 / 25.4 * dpi), int(297 / 25.4 * dpi)
    bw, bh = (cols + 1) * px, (rows + 1) * px
    if bw > w and bw <= h:          # wider than portrait A4: use landscape
        w, h = h, w
    img = np.full((h, w), 255, np.uint8)
    if bw > w or bh > h:
        raise SystemExit("board does not fit on A4: use smaller squares")
    x0, y0 = (w - bw) // 2, (h - bh) // 2
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                img[y0 + r * px:y0 + (r + 1) * px, x0 + c * px:x0 + (c + 1) * px] = 0
    cv2.putText(img, f"{cols}x{rows} inner corners, {square_mm:g} mm squares - print at 100%",
                (x0, y0 + bh + 80), cv2.FONT_HERSHEY_SIMPLEX, 1.6, 0, 3)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    # PNG dpi tag so most viewers print it at the right size
    print(f"wrote {path}  ({cols + 1}x{rows + 1} squares of {square_mm:g} mm). "
          f"Print at 100% and MEASURE a square afterwards.")


def find(gray, size):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    ok, corners = cv2.findChessboardCorners(gray, size, flags)
    if not ok:
        return None
    cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1),
                     (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01))
    return corners


def describe(corners, shape):
    """Where the board is in the picture: centre (fraction) and size (fraction)."""
    p = corners.reshape(-1, 2)
    h, w = shape
    c = p.mean(axis=0) / (w, h)
    span = (p.max(axis=0) - p.min(axis=0)) / (w, h)
    g = corners.reshape(ROWS, COLS, 2)
    left, right = np.linalg.norm(g[0, 0] - g[-1, 0]), np.linalg.norm(g[0, -1] - g[-1, -1])
    top, bot = np.linalg.norm(g[0, 0] - g[0, -1]), np.linalg.norm(g[-1, 0] - g[-1, -1])
    tlr = (left - right) / max(left, right)          # one side further away than the other
    tud = (top - bot) / max(top, bot)
    return np.array([c[0], c[1], span[0], span[1], tlr, tud])


def is_new(desc, taken, min_diff=0.08):
    return all(np.abs(desc - d).max() > min_diff for d in taken)


COLS, ROWS = 9, 6      # set from the command line in main()
STILL_PX, STILL_FRAMES = 0.6, 3     # corners moved < 0.6 px for 3 frames in a row = held still
MAX_SHOTS = 45


def calibrate(objp, shots, shape):
    h, w = shape
    obj = [objp] * len(shots)
    # calibrateCameraExtended reports each picture's own error (OpenCV's numbers).
    rms, K, dist, rvecs, tvecs, _, _, per_view = cv2.calibrateCameraExtended(obj, shots, (w, h), None, None)
    per = [float(e) for e in np.ravel(per_view)]
    return rms, K, dist.ravel(), per, [np.array(r) for r in rvecs], [np.array(t) for t in tvecs]


def robust_calibrate(objp, shots, shape):
    """Calibrate, drop pictures that fit much worse than the rest, repeat."""
    keep = list(range(len(shots)))
    dropped = []
    for _ in range(3):
        rms, K, dist, per, rv, tv = calibrate(objp, [shots[i] for i in keep], shape)
        lim = max(1.5, 4 * float(np.median(per)))
        bad = [k for k, e in zip(keep, per) if e > lim]
        if not bad or len(keep) - len(bad) < 8:
            break
        dropped += bad
        keep = [k for k in keep if k not in bad]
    return rms, K, dist, per, rv, tv, keep, dropped


def variety(rvecs, tvecs):
    """Board distances (m) and tilts (deg from facing the camera) over the pictures."""
    dists = [float(np.linalg.norm(t)) for t in tvecs]
    tilts = []
    for r in rvecs:
        R, _ = cv2.Rodrigues(r)
        tilts.append(float(np.degrees(np.arccos(min(1.0, abs(R[2, 2]))))))
    return dists, tilts


def checklist(descs):
    """What the pictures so far cover: 3x3 grid cells, close, far, tilts (from the
    board's outline only, no calibration needed)."""
    cells = {(min(2, int(d[0] * 3)), min(2, int(d[1] * 3))) for d in descs}
    close = any(max(d[2], d[3]) > 0.6 for d in descs)
    far = any(max(d[2], d[3]) < 0.3 for d in descs)
    # tilts counted per direction: need both ways (min of the two sides)
    tilt_lr = min(sum(d[4] > 0.12 for d in descs), sum(d[4] < -0.12 for d in descs))
    tilt_ud = min(sum(d[5] > 0.12 for d in descs), sum(d[5] < -0.12 for d in descs))
    return cells, close, far, tilt_lr, tilt_ud


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--make-board", action="store_true", help="write a printable checkerboard PNG and exit")
    ap.add_argument("--cols", type=int, default=9, help="inner corners across")
    ap.add_argument("--rows", type=int, default=6, help="inner corners down")
    ap.add_argument("--square", type=float, default=None, help="measured square size in METRES (e.g. 0.025)")
    ap.add_argument("--square-mm", type=float, default=25.0, help="square size for --make-board")
    ap.add_argument("--port", default="COM11")
    ap.add_argument("--shots", type=int, default=20, help="calibrate automatically after this many pictures")
    ap.add_argument("--exposure", type=int, default=DEFAULT_EXPOSURE)
    ap.add_argument("--out", type=Path, default=HERE / "calib_xiao.yaml")
    ap.add_argument("--load", type=Path, default=None,
                    help="recalibrate from saved pictures (calib_xiao_shots.npz) without the camera")
    args = ap.parse_args()

    size = (args.cols, args.rows)
    global COLS, ROWS
    COLS, ROWS = args.cols, args.rows
    if args.make_board:
        make_board(args.cols, args.rows, args.square_mm, HERE / f"checkerboard_{args.cols}x{args.rows}_{args.square_mm:g}mm.png")
        return
    if args.square is None:
        raise SystemExit("give the measured square size: --square 0.025  (metres)")

    objp = np.zeros((args.cols * args.rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2) * args.square
    raw_path = args.out.with_name(args.out.stem + "_shots.npz")   # every picture, saved as taken

    if args.load:
        z = np.load(args.load)
        shots = [s_ for s_ in z["shots"]]
        shape = tuple(int(v) for v in z["shape"])
        print(f"loaded {len(shots)} pictures from {args.load.name} (square {args.square * 1000:.1f} mm)")
        report(args, objp, shots, shape, robust_calibrate(objp, shots, shape))
        return

    ser = serial.Serial(args.port, 115200, timeout=0.2)
    reader = Reader(ser)
    reader.start()
    time.sleep(0.3)
    set_camera(ser, args.exposure)

    shots, descs, last, shape, result = [], [], 0, None, None
    prev, still = None, 0          # board must be held still for STILL_FRAMES frames
    print(f"Looking for a {args.cols}x{args.rows} board ({args.square * 1000:.1f} mm squares). "
          f"Move it around; {args.shots} pictures wanted.")
    try:
        while True:
            frame, _ = reader.latest()
            key = cv2.waitKey(5) & 0xFF
            if key == 27:                                  # Esc: quit, no calibration
                break
            if key == ord("q"):                            # q: calibrate with what we have
                if len(shots) >= 8:
                    result = robust_calibrate(objp, shots, shape)
                break
            if key == ord("u") and shots:
                shots.pop(); descs.pop()
                print(f"  removed last picture ({len(shots)} left)")
            if not frame or frame[0] == last:
                if key == ord("c") and len(shots) >= 8:
                    result = robust_calibrate(objp, shots, shape)
                    break
                continue
            last, gray = frame
            shape = gray.shape
            corners = find(gray, size)
            view = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            if corners is not None:
                cv2.drawChessboardCorners(view, size, corners, True)
                d = describe(corners, shape)
                move = 99.0 if prev is None else float(np.abs(corners - prev).max())
                still = still + 1 if move < STILL_PX else 0
                prev = corners.copy()
                if still < STILL_FRAMES and key != ord(" "):
                    cv2.putText(view, "hold still", (4, 236), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
                elif key == ord(" ") or is_new(d, descs):
                    shots.append(corners.copy()); descs.append(d)
                    np.savez(raw_path, shots=np.array(shots), objp=objp, shape=np.array(gray.shape))
                    view[:] = 255 - view                     # flash = picture taken
                    print(f"  picture {len(shots)}/{args.shots}  centre ({d[0]:.2f},{d[1]:.2f})  size {d[2]:.2f}"
                          f"  tilt L/R {d[4]:+.2f} U/D {d[5]:+.2f}")
            big = cv2.resize(view, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST)
            cells, close, far, tlr, tud = checklist(descs)
            H, W = big.shape[:2]
            for i in (1, 2):
                cv2.line(big, (W * i // 3, 0), (W * i // 3, H), (90, 90, 90), 1)
                cv2.line(big, (0, H * i // 3), (W, H * i // 3), (90, 90, 90), 1)
            for (cx_, cy_) in cells:
                cv2.circle(big, (W * cx_ // 3 + W // 6, H * cy_ // 3 + H // 6), 8, (0, 200, 0), -1)
            todo = [n for n, ok in (("CLOSE (fill the picture)", close), ("FAR (small)", far),
                                    (f"tilt BOTH ways left/right {tlr}/2", tlr >= 2), (f"tilt BOTH ways up/down {tud}/2", tud >= 2),
                                    (f"grid {len(cells)}/9", len(cells) >= 8)) if not ok]
            cv2.putText(big, "still need: " + (", ".join(todo) if todo else "nothing - press c"),
                        (8, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)
            cv2.putText(big, f"{len(shots)}/{args.shots} pictures  -  "
                        f"{'board found' if corners is not None else 'no board'}  -  c calibrate, q quit",
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0) if corners is not None else (0, 0, 255), 1)
            cv2.imshow("cam_calib", big)
            if corners is None:
                prev, still = None, 0
            done = not todo and len(shots) >= args.shots
            if key == ord("c") and len(shots) >= 8 or done or len(shots) >= MAX_SHOTS:
                result = robust_calibrate(objp, shots, shape)
                break
    except KeyboardInterrupt:
        if len(shots) >= 8:
            result = robust_calibrate(objp, shots, shape)
    finally:
        reader.running = False
        ser.close()
        cv2.destroyAllWindows()

    if result is None:
        print(f"Not calibrated ({len(shots)} pictures; need at least 8)."
              + (f" Pictures saved: python cam_calib.py --square {args.square} --load {raw_path.name}" if len(shots) >= 8 else ""))
        return
    report(args, objp, shots, shape, result)


def report(args, objp, shots, shape, result):
    rms, K, dist, per, rvecs, tvecs, keep, dropped = result
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    ah = load_apriltag_host(DEFAULT_REPO)
    D = ah.FIRMWARE_INTRINSICS
    print(f"\nCALIBRATION  ({len(keep)} of {len(shots)} pictures used, image {shape[1]}x{shape[0]})")
    if dropped:
        print(f"  dropped pictures {[i + 1 for i in dropped]} (fitted much worse than the rest)")
    per_rms = float(np.sqrt(np.mean(np.square(per))))
    print(f"  reprojection error  {rms:.3f} px   (under ~0.5 px is good; worst picture {max(per):.2f} px"
          f"{'' if abs(per_rms - rms) < 0.2 + 0.2 * rms else f'; per-picture check gives {per_rms:.2f} - inconsistent, please send me the _shots.npz file'})")
    print(f"  fx {fx:8.2f}   fy {fy:8.2f}   cx {cx:7.2f}   cy {cy:7.2f}")
    print(f"  distortion k1 {dist[0]:+.4f}  k2 {dist[1]:+.4f}  p1 {dist[2]:+.4f}  p2 {dist[3]:+.4f}  k3 {dist[4]:+.4f}")
    print(f"  vs drone numbers: fx {D.fx:.2f} (x{fx / D.fx:.3f})  fy {D.fy:.2f} (x{fy / D.fy:.3f})  "
          f"cx {D.cx:.1f}  cy {D.cy:.1f}")
    corner = np.array([[[0, 0]], [[shape[1] - 1, 0]]], np.float32)
    und = cv2.undistortPoints(corner, K, dist, P=K).reshape(-1, 2)
    print(f"  distortion moves the image corner by {np.hypot(*(und[0] - corner[0, 0])):.1f} px")
    dists, tilts = variety(rvecs, tvecs)
    print(f"  board distance {min(dists):.2f} - {max(dists):.2f} m,  tilt {min(tilts):.0f} - {max(tilts):.0f} deg")
    warn = []
    if max(dists) < 1.8 * min(dists):
        warn.append("distances too similar: add pictures much CLOSER and much FURTHER")
    if max(tilts) < 30:
        warn.append("not enough tilt: add pictures with the board tilted 30-45 deg")
    if rms > 1.0:
        warn.append(f"error {rms:.2f} px is high (blur / board moving / board not flat)")
    if abs(fx / fy - 1) > 0.05:
        warn.append(f"fx and fy differ by {abs(fx / fy - 1) * 100:.0f}% (expect <5%) - result is shaky")
    for w_ in warn:
        print("  WARNING: " + w_)
    if warn:
        print("  -> the focal length is not reliable yet; re-run with more variety")
    out = dict(image_size=[int(shape[1]), int(shape[0])], rms_px=round(float(rms), 4),
               intrinsics=dict(fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy)),
               distortion=[float(v) for v in dist[:5]],
               board=dict(cols=args.cols, rows=args.rows, square_m=args.square, pictures=len(keep)),
               board_distance_m=[round(min(dists), 3), round(max(dists), 3)], max_tilt_deg=round(max(tilts), 1),
               warnings=warn,
               made=time.strftime("%Y-%m-%d %H:%M"))
    args.out.write_text(yaml.safe_dump(out, sort_keys=False))
    print(f"\nsaved {args.out.name}.  loc_test.py uses it when loc_arena.yaml has:  calibration: {args.out.name}")


if __name__ == "__main__":
    main()
