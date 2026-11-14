# Camera calibration

Checkerboard calibration of the drone camera (320x240 grayscale). It produced the
`F_X`, `F_Y`, `C_X`, `C_Y` values in `main/at_detect.c`. The result is saved in
`calib_drone3.yaml`: fx 188.9, fy 188.2, centre (160.5, 118.1), RMS 0.30 px, 19 pictures.

The camera is streamed over USB serial by a XIAO ESP32-S3 running `xiao_firmware/`
(PlatformIO, COM11 by default).

```bash
python cam_calib.py --make-board      # print checkerboard_9x6_25mm.png at 100 %
python cam_calib.py --square 0.024    # measured square size in metres
```

Hold the board still in new positions (near/far, all corners, tilted both ways).
A picture is taken automatically; it calibrates after 20 pictures (c = now, Esc = quit).
`tag_test.py` holds the shared serial reader and runs the drone's AprilTag detector live.
