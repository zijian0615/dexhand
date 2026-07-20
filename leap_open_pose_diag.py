"""LEAP Hand open-pose motor diagnostics without Quest teleop."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

LEAP_API_DIR = Path(__file__).resolve().parents[1] / "LEAP_Hand_API" / "python"
if str(LEAP_API_DIR) not in sys.path:
    sys.path.insert(0, str(LEAP_API_DIR))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Command LEAP open pose and read selected motor states."
    )
    parser.add_argument(
        "--motors",
        type=int,
        nargs="+",
        default=[5, 6, 7],
        help="Motor IDs to print. Default: 5 6 7 (middle curl joints).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Seconds to keep reading after commanding open pose.",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=5.0,
        help="Read/print rate in Hz.",
    )
    parser.add_argument(
        "--curr-lim",
        type=int,
        default=350,
        help="LEAP current limit in mA before commanding open pose.",
    )
    parser.add_argument(
        "--kp",
        type=int,
        default=None,
        help="Optional position P gain. Use lower values like 250 for gentle tests.",
    )
    parser.add_argument(
        "--kd",
        type=int,
        default=None,
        help="Optional position D gain.",
    )
    parser.add_argument(
        "--no-command",
        action="store_true",
        help="Only read motor states after LeapNode init; do not re-send open pose.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Also print Dynamixel status registers for the selected motors.",
    )
    parser.add_argument(
        "--registers",
        action="store_true",
        help="Also print raw goal/present position and current limit registers.",
    )
    parser.add_argument(
        "--direct-write",
        action="store_true",
        help="After the normal group command, write Goal Position to selected motors one-by-one and verify it.",
    )
    return parser.parse_args()


def _read_register(client, motor_id: int, address: int, size: int):
    if size == 1:
        value, comm_result, dxl_error = client.packet_handler.read1ByteTxRx(
            client.port_handler, motor_id, address
        )
    elif size == 2:
        value, comm_result, dxl_error = client.packet_handler.read2ByteTxRx(
            client.port_handler, motor_id, address
        )
    elif size == 4:
        value, comm_result, dxl_error = client.packet_handler.read4ByteTxRx(
            client.port_handler, motor_id, address
        )
    else:
        raise ValueError(f"Unsupported register size: {size}")
    ok = client.handle_packet_result(
        comm_result,
        dxl_error,
        motor_id,
        context=f"read_register@{address}",
    )
    return value if ok else None


def _format_status(hand, motors):
    client = hand.dxl_client
    rows = []
    for motor_id in motors:
        mode = _read_register(client, motor_id, 11, 1)
        torque = _read_register(client, motor_id, 64, 1)
        shutdown = _read_register(client, motor_id, 63, 1)
        hw_error = _read_register(client, motor_id, 70, 1)
        voltage_raw = _read_register(client, motor_id, 144, 2)
        temp = _read_register(client, motor_id, 146, 1)
        voltage = None if voltage_raw is None else voltage_raw / 10.0
        rows.append(
            f"{motor_id}:mode={mode},torque={torque},shutdown={shutdown},"
            f"hwerr={hw_error},volt={voltage}V,temp={temp}C"
        )
    return "status=[" + "; ".join(rows) + "]"


def _format_registers(hand, motors, target=None):
    client = hand.dxl_client
    pos_scale = client._pos_vel_cur_reader.pos_scale
    rows = []
    for motor_id in motors:
        goal_raw = _read_register(client, motor_id, 116, 4)
        present_raw = _read_register(client, motor_id, 132, 4)
        current_limit_raw = _read_register(client, motor_id, 38, 2)
        goal_current_raw = _read_register(client, motor_id, 102, 2)
        goal_rad = None if goal_raw is None else goal_raw * pos_scale
        present_rad = None if present_raw is None else present_raw * pos_scale
        current_limit = (
            None if current_limit_raw is None else current_limit_raw * 1.34
        )
        expected_raw = None
        if target is not None and 0 <= motor_id < len(target):
            expected_raw = int(target[motor_id] / pos_scale)
        goal_matches = None if expected_raw is None or goal_raw is None else goal_raw == expected_raw
        rows.append(
            f"{motor_id}:goal_raw={goal_raw},goal_rad={goal_rad},"
            f"present_raw={present_raw},present_rad={present_rad},"
            f"expected_goal_raw={expected_raw},goal_matches={goal_matches},"
            f"current_limit_raw={current_limit_raw},current_limit={current_limit}mA,"
            f"goal_current_raw={goal_current_raw},goal_current="
            f"{None if goal_current_raw is None else goal_current_raw * 1.34}mA"
        )
    return "registers=[" + "; ".join(rows) + "]"


def _format_rows(target, pos, vel, cur, motors):
    rows = []
    for motor_id in motors:
        if motor_id < 0 or motor_id >= len(target):
            rows.append(f"{motor_id}:out-of-range")
            continue
        err = target[motor_id] - pos[motor_id]
        rows.append(
            f"{motor_id}:t={target[motor_id]:.3f},p={pos[motor_id]:.3f},"
            f"e={err:.3f},v={vel[motor_id]:.3f},cur={cur[motor_id]:.1f}mA"
        )
    return "; ".join(rows)


def _direct_write_targets(hand, motors, target, current_limit_ma):
    """Write Goal Position individually, so each motor returns an acknowledgement."""
    client = hand.dxl_client
    pos_scale = client._pos_vel_cur_reader.pos_scale
    cur_scale = client._pos_vel_cur_reader.cur_scale
    for motor_id in motors:
        if motor_id < 0 or motor_id >= len(target):
            print(f"direct_write={motor_id}:out-of-range")
            continue
        raw_current = round(current_limit_ma / cur_scale)
        comm_result, dxl_error = client.packet_handler.write2ByteTxRx(
            client.port_handler, motor_id, 102, raw_current
        )
        current_ok = client.handle_packet_result(
            comm_result,
            dxl_error,
            motor_id,
            context="direct_write_goal_current",
        )
        current_readback = _read_register(client, motor_id, 102, 2) if current_ok else None
        raw_target = int(target[motor_id] / pos_scale)
        value = raw_target.to_bytes(4, byteorder="little", signed=False)
        comm_result, dxl_error = client.packet_handler.writeTxRx(
            client.port_handler, motor_id, 116, 4, value
        )
        ok = client.handle_packet_result(
            comm_result,
            dxl_error,
            motor_id,
            context="direct_write_goal_position",
        )
        goal_raw = _read_register(client, motor_id, 116, 4) if ok else None
        print(
            f"direct_write={motor_id}:goal_current_raw={raw_current},"
            f"goal_current_readback={current_readback},"
            f"target_raw={raw_target},"
            f"readback_raw={goal_raw},matches={goal_raw == raw_target}"
        )


def main():
    args = parse_args()

    from main import LeapNode

    hand = LeapNode()
    hand.curr_lim = args.curr_lim
    hand.dxl_client.sync_write(
        hand.motors,
        np.ones(len(hand.motors)) * args.curr_lim,
        102,
        2,
    )
    if args.kp is not None:
        hand.kP = args.kp
        hand.dxl_client.sync_write(hand.motors, np.ones(len(hand.motors)) * args.kp, 84, 2)
        hand.dxl_client.sync_write([0, 4, 8], np.ones(3) * (args.kp * 0.75), 84, 2)
    if args.kd is not None:
        hand.kD = args.kd
        hand.dxl_client.sync_write(hand.motors, np.ones(len(hand.motors)) * args.kd, 80, 2)
        hand.dxl_client.sync_write([0, 4, 8], np.ones(3) * (args.kd * 0.75), 80, 2)

    if not args.no_command:
        hand.set_allegro(np.zeros(16))
        print("Requested open pose; verify goal_matches below before judging motor motion.")
    else:
        print("Reading only; not re-sending open pose.")

    motors = sorted(set(args.motors))
    if args.direct_write:
        _direct_write_targets(hand, motors, hand.curr_pos, args.curr_lim)
    print(f"Diagnostics motors={motors}, curr_lim={args.curr_lim}mA")
    print("Fields: t=target rad, p=actual rad, e=target-actual rad, v=rad/s, cur=mA")
    if args.status:
        print(_format_status(hand, motors))
    if args.registers:
        print(_format_registers(hand, motors, hand.curr_pos))

    period = 1.0 / max(args.rate, 1e-6)
    deadline = time.monotonic() + max(args.duration, 0.0)
    sample = 0
    try:
        while time.monotonic() <= deadline:
            pos, vel, cur = hand.pos_vel_eff_srv()
            target = np.asarray(hand.curr_pos, dtype=float)
            pos = np.asarray(pos, dtype=float)
            vel = np.asarray(vel, dtype=float)
            cur = np.asarray(cur, dtype=float)
            print(f"sample={sample} {_format_rows(target, pos, vel, cur, motors)}")
            if args.status:
                print(_format_status(hand, motors))
            if args.registers:
                print(_format_registers(hand, motors, target))
            sample += 1
            time.sleep(period)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
