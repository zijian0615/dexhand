"""Quest HTS -> PyBullet IK -> LEAP Hand real robot teleop.

Uses the Bidex fingertip-matching inverse kinematics in avp_leap.py instead of
the heuristic curl mapping. PyBullet is required, so run this on Linux/Jetson
(PyBullet does not build on the user's macOS).

Typical use on Jetson:
    # forward the HTS TCP port from the Quest (USB): adb reverse tcp:8000 tcp:8000
    python Bidex_VisionPro_Teleop/quest_leap_ik.py --port 8000 --curr-lim 300 --no-gui
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
from hand_tracking_sdk import HTSClient, HTSClientConfig, StreamOutput, TransportMode
from hand_tracking_sdk.models import HandSide

from avp_leap import Leapv1PybulletIKPython
from quest_hand_pos import frame_matches_hand

LEAP_API_DIR = Path(__file__).resolve().parents[1] / "LEAP_Hand_API" / "python"
if str(LEAP_API_DIR) not in sys.path:
    sys.path.insert(0, str(LEAP_API_DIR))


def parse_args():
    parser = argparse.ArgumentParser(description="Quest HTS -> IK -> LEAP Hand teleop")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--left", action="store_true", help="Use left hand (HTS + left URDF)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run IK only, do not command the robot",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Run PyBullet headless (use on a Jetson with no display)",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.3,
        help="Exponential smoothing on IK output [0=off, 1=instant].",
    )
    parser.add_argument(
        "--no-latest-frame",
        action="store_true",
        help="Process every queued frame instead of keeping only the newest",
    )
    parser.add_argument("--curr-lim", type=int, default=350, help="LEAP current limit (mA)")
    parser.add_argument("--kp", type=int, default=None, help="Position P gain (stiffness)")
    parser.add_argument("--kd", type=int, default=None, help="Position D gain (damping)")
    parser.add_argument(
        "--skip-motors",
        type=int,
        nargs="+",
        default=None,
        metavar="ID",
        help="Motor IDs (0-15) to keep at open pose (e.g. a faulty motor).",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _enqueue_latest(q: queue.Queue, item) -> None:
    try:
        q.put_nowait(item)
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        q.put_nowait(item)


def _latest_frame_stream(client: HTSClient, stop: threading.Event):
    q: queue.Queue = queue.Queue(maxsize=1)

    def pump() -> None:
        try:
            for frame in client.iter_events():
                if stop.is_set():
                    break
                _enqueue_latest(q, frame)
        finally:
            stop.set()

    threading.Thread(target=pump, daemon=True).start()
    while not stop.is_set():
        try:
            yield q.get(timeout=0.1)
        except queue.Empty:
            continue


def main():
    args = parse_args()
    use_latest_frame = not args.no_latest_frame
    skip_motors = sorted(set(args.skip_motors)) if args.skip_motors else []
    expected_side = HandSide.LEFT if args.left else HandSide.RIGHT
    is_left = args.left

    pbik = Leapv1PybulletIKPython(is_left=is_left, use_avp=False, gui=not args.no_gui)
    print(f"PyBullet IK ready (gui={not args.no_gui}, {'left' if is_left else 'right'} hand).")

    hand = None
    if not args.dry_run:
        from main import LeapNode

        hand = LeapNode()
        hand.curr_lim = args.curr_lim
        hand.dxl_client.sync_write(hand.motors, np.ones(len(hand.motors)) * args.curr_lim, 102, 2)
        if args.kp is not None:
            hand.kP = args.kp
            hand.dxl_client.sync_write(hand.motors, np.ones(len(hand.motors)) * args.kp, 84, 2)
            hand.dxl_client.sync_write([0, 4, 8], np.ones(3) * (args.kp * 0.75), 84, 2)
        if args.kd is not None:
            hand.kD = args.kd
            hand.dxl_client.sync_write(hand.motors, np.ones(len(hand.motors)) * args.kd, 80, 2)
            hand.dxl_client.sync_write([0, 4, 8], np.ones(3) * (args.kd * 0.75), 80, 2)
        hand.set_allegro(np.zeros(16))
        print(f"LEAP Hand connected. curr_lim={args.curr_lim} mA")
        if skip_motors:
            print(f"Skipping (held open) motors: {skip_motors}")
    else:
        print("Dry run: IK output only, robot will not move.")

    client = HTSClient(
        HTSClientConfig(
            transport_mode=TransportMode.TCP_SERVER,
            host=args.host,
            port=args.port,
            output=StreamOutput.FRAMES,
        )
    )

    print(f"Listening on tcp://{args.host}:{args.port} ({expected_side.value} hand)...")
    print("Move your hand in front of the Quest cameras. Ctrl+C to stop.\n")

    stop = threading.Event()
    frame_stream = (
        _latest_frame_stream(client, stop) if use_latest_frame else client.iter_events()
    )
    prev = None
    alpha = float(np.clip(args.smoothing, 0.0, 1.0))
    sent = 0
    try:
        for frame in frame_stream:
            if frame.side != expected_side or not frame_matches_hand(frame, is_left):
                continue

            joints = pbik.get_quest_data(frame)
            if joints is None:
                continue
            joints = np.asarray(joints, dtype=float)

            if alpha > 0.0:
                prev = joints if prev is None else (1.0 - alpha) * prev + alpha * joints
                joints = prev

            if skip_motors:
                joints[skip_motors] = 0.0

            if hand is not None:
                hand.set_allegro(joints)

            sent += 1
            if args.verbose and sent % 30 == 0:
                print(f"cmd={sent} joints={np.round(joints, 3).tolist()}")
    except OSError as exc:
        if exc.errno == 48:
            raise SystemExit(
                f"Port {args.port} is already in use. Stop the other listener first:\n"
                f"  lsof -nP -iTCP:{args.port} -sTCP:LISTEN\n"
                f"  kill <PID>"
            ) from exc
        raise
    except KeyboardInterrupt:
        stop.set()
        if hand is not None:
            hand.set_allegro(np.zeros(16))
            time.sleep(0.3)
        print(f"\nStopped. Sent {sent} commands.")


if __name__ == "__main__":
    main()
