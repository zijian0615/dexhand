"""Verify Quest HTS connection without PyBullet."""

import argparse
import time

from hand_tracking_sdk import HTSClient, HTSClientConfig, StreamOutput, TransportMode
from hand_tracking_sdk.models import HandSide, JointName

from quest_hand_pos import quest_frame_to_hand_pos


def parse_args():
    parser = argparse.ArgumentParser(description="Test Quest HTS hand tracking stream")
    parser.add_argument("--host", default="0.0.0.0", help="TCP bind host on this machine")
    parser.add_argument("--port", type=int, default=8000, help="TCP port configured in HTS app")
    parser.add_argument(
        "--left",
        action="store_true",
        help="Only print Left hand frames",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    expected_side = HandSide.LEFT if args.left else HandSide.RIGHT

    client = HTSClient(
        HTSClientConfig(
            transport_mode=TransportMode.TCP_SERVER,
            host=args.host,
            port=args.port,
            output=StreamOutput.FRAMES,
        )
    )

    print(f"Listening on tcp://{args.host}:{args.port} for {expected_side.value} hand...")
    print("Move your hand in Quest. Ctrl+C to stop.\n")

    frame_count = 0
    try:
        for frame in client.iter_events():
            if frame.side != expected_side:
                continue

            frame_count += 1
            tip = frame.get_joint(JointName.INDEX_TIP)
            hand_pos = quest_frame_to_hand_pos(frame)

            if frame_count % 30 == 0:
                print(
                    f"frame={frame_count} seq={frame.sequence_id} "
                    f"index_tip=({tip[0]:.3f}, {tip[1]:.3f}, {tip[2]:.3f}) "
                    f"hand_pos[3]=({hand_pos[3][0]:.3f}, {hand_pos[3][1]:.3f}, {hand_pos[3][2]:.3f})"
                )
            time.sleep(0.001)
    except OSError as exc:
        if exc.errno == 48:
            raise SystemExit(
                f"Port {args.port} is already in use. Stop the other listener first:\n"
                f"  lsof -nP -iTCP:{args.port} -sTCP:LISTEN\n"
                f"  kill <PID>"
            ) from exc
        raise
    except KeyboardInterrupt:
        print(f"\nStopped. Received {frame_count} frames.")


if __name__ == "__main__":
    main()
