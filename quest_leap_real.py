"""Quest HTS -> LEAP Hand real robot teleop (no PyBullet required)."""

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

from quest_hand_pos import frame_matches_hand
from quest_leap_retarget import LeapRetargetConfig, QuestLeapRetargeter

LEAP_API_DIR = Path(__file__).resolve().parents[1] / "LEAP_Hand_API" / "python"
if str(LEAP_API_DIR) not in sys.path:
    sys.path.insert(0, str(LEAP_API_DIR))


def parse_args():
    parser = argparse.ArgumentParser(description="Quest HTS -> LEAP Hand teleop")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--left", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print joint targets without commanding the robot",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=0.0,
        help="Optional max command rate in Hz (0 = send every latest frame)",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.15,
        help="Exponential smoothing [0=off, 1=instant]. Lower = less lag.",
    )
    parser.add_argument(
        "--low-latency",
        action="store_true",
        help="Low-latency preset: smoothing=0.08, latest-frame-only queue",
    )
    parser.add_argument(
        "--no-latest-frame",
        action="store_true",
        help="Process every queued frame instead of keeping only the newest",
    )
    parser.add_argument(
        "--calibrate-frames",
        type=int,
        default=45,
        help="Average this many open-hand frames at start as zero pose",
    )
    parser.add_argument(
        "--fist-sync",
        type=float,
        default=None,
        help="Blend weak fingers toward strongest curl [0-1], default 0.50",
    )
    parser.add_argument(
        "--fist-follow",
        type=float,
        default=None,
        help="Min fraction of strongest curl applied to weak fingers, default 0.85",
    )
    parser.add_argument("--curr-lim", type=int, default=350, help="LEAP current limit (mA)")
    parser.add_argument(
        "--kp",
        type=int,
        default=None,
        help="Position P gain (stiffness). Default 600; try 250-350 to avoid stall overload.",
    )
    parser.add_argument(
        "--kd",
        type=int,
        default=None,
        help="Position D gain (damping). Default 200.",
    )
    parser.add_argument(
        "--max-joint-rad",
        type=float,
        default=None,
        help="Cap on commanded curl per joint (rad). Lower = less chance of stall overload.",
    )
    parser.add_argument(
        "--skip-motors",
        type=int,
        nargs="+",
        default=None,
        metavar="ID",
        help="Motor IDs (0-15) to keep at open pose, e.g. --skip-motors 6 (a faulty motor).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print raw curl deltas (helps debug tracking/ calibration)",
    )
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
    if args.low_latency:
        args.smoothing = 0.08
    use_latest_frame = not args.no_latest_frame
    skip_motors = sorted(set(args.skip_motors)) if args.skip_motors else []
    expected_side = HandSide.LEFT if args.left else HandSide.RIGHT
    is_left = args.left

    retarget_cfg = LeapRetargetConfig(
        smoothing=args.smoothing,
        calibrate_frames=args.calibrate_frames,
    )
    if args.fist_sync is not None:
        retarget_cfg.fist_sync = args.fist_sync
    if args.fist_follow is not None:
        retarget_cfg.fist_follow = args.fist_follow
    if args.max_joint_rad is not None:
        retarget_cfg.max_joint_rad = args.max_joint_rad
    retargeter = QuestLeapRetargeter(retarget_cfg)

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
        kp_note = args.kp if args.kp is not None else hand.kP
        print(f"LEAP Hand connected. curr_lim={args.curr_lim} mA, kP={kp_note}")
        if skip_motors:
            print(f"Skipping (held open) motors: {skip_motors}")
    else:
        print("Dry run: joint targets only, robot will not move.")

    client = HTSClient(
        HTSClientConfig(
            transport_mode=TransportMode.TCP_SERVER,
            host=args.host,
            port=args.port,
            output=StreamOutput.FRAMES,
        )
    )

    print(f"Listening on tcp://{args.host}:{args.port} ({expected_side.value} hand)...")
    latency_mode = "latest-frame" if use_latest_frame else "fifo"
    rate_note = f"{args.rate:g} Hz cap" if args.rate > 0 else "unlimited"
    print(f"Latency: {latency_mode}, smoothing={args.smoothing}, command rate={rate_note}")
    print(
        f"Keep your hand OPEN and still for ~{args.calibrate_frames / 30:.1f}s "
        "to calibrate zero pose."
    )
    print("Ctrl+C to stop.\n")

    min_period = 0.0 if args.rate <= 0 else 1.0 / args.rate
    stop = threading.Event()
    frame_stream = (
        _latest_frame_stream(client, stop)
        if use_latest_frame
        else client.iter_events()
    )
    sent = 0
    calibrated_announced = False
    last_cmd_t = 0.0
    log_every = 30
    try:
        for frame in frame_stream:
            if frame.side != expected_side or not frame_matches_hand(frame, is_left):
                continue

            if min_period > 0:
                now = time.monotonic()
                wait = min_period - (now - last_cmd_t)
                if wait > 0:
                    time.sleep(wait)

            joints = retargeter.retarget(frame)
            sent += 1
            last_cmd_t = time.monotonic()

            if not retargeter.calibrated:
                if sent == 1 or sent % log_every == 0:
                    pct = int(retargeter.calibrate_progress * 100)
                    print(f"calibrating... {pct}%  (keep hand open and still)")
                continue

            if not calibrated_announced:
                print("Calibration complete. Now curl fingers in front of Quest cameras.")
                if args.verbose:
                    print(f"baseline curl rad={retargeter.baseline_summary()}")
                calibrated_announced = True

            if skip_motors:
                joints[skip_motors] = 0.0

            if hand is not None:
                hand.set_allegro(joints)

            if sent % log_every == 0:
                line = f"cmd={sent} joints={np.round(joints, 3).tolist()}"
                if args.verbose:
                    deltas = retargeter.debug_curl_deltas(frame)
                    line += f" curl_d={deltas}"
                print(line)

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
