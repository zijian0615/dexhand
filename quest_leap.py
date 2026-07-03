"""Quest 3 + HTS teleop demo for LEAP Hand PyBullet IK."""

import argparse
import importlib.util
import sys
import time


def _has_pybullet() -> bool:
    return importlib.util.find_spec("pybullet") is not None


def parse_args():
    parser = argparse.ArgumentParser(description="Quest HTS -> LEAP Hand IK demo")
    parser.add_argument("--host", default="0.0.0.0", help="TCP bind host on this machine")
    parser.add_argument("--port", type=int, default=8000, help="TCP port configured in HTS app")
    parser.add_argument(
        "--left",
        action="store_true",
        help="Use left-hand LEAP URDF and expect HTS Left frames",
    )
    parser.add_argument(
        "--stream-only",
        action="store_true",
        help="Only verify HTS stream (no PyBullet IK)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.stream_only or not _has_pybullet():
        if not args.stream_only and not _has_pybullet():
            print(
                "PyBullet is not installed on this machine (common on macOS).\n"
                "Running stream test instead. Use quest_leap.py --stream-only explicitly.\n"
            )
        from quest_stream_test import main as stream_main

        sys.argv = [
            "quest_stream_test.py",
            "--host",
            args.host,
            "--port",
            str(args.port),
        ]
        if args.left:
            sys.argv.append("--left")
        stream_main()
        return

    from avp_leap import Leapv1PybulletIKPython
    from hand_tracking_sdk import HTSClient, HTSClientConfig, StreamOutput, TransportMode
    from hand_tracking_sdk.models import HandSide

    expected_side = HandSide.LEFT if args.left else HandSide.RIGHT

    pbik = Leapv1PybulletIKPython(is_left=args.left, use_avp=False)
    client = HTSClient(
        HTSClientConfig(
            transport_mode=TransportMode.TCP_SERVER,
            host=args.host,
            port=args.port,
            output=StreamOutput.FRAMES,
        )
    )

    print(f"Waiting for Quest HTS on tcp://{args.host}:{args.port} ({expected_side.value} hand)...")
    try:
        for frame in client.iter_events():
            if frame.side != expected_side:
                continue
            joints = pbik.get_quest_data(frame)
            if joints is None:
                continue
            print(joints)
            time.sleep(0.03)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
