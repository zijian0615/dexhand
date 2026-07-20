"""Safely pulse one LEAP Hand motor to identify its physical joint.

The script moves only from the current position by a small amount, restores the
initial target, and then disables torque.

Example:
    uv run python Bidex_VisionPro_Teleop/test_motor_id6_motion.py
"""

from __future__ import annotations

import argparse
import glob
import sys
import time

from dynamixel_sdk import COMM_SUCCESS, PacketHandler, PortHandler

BAUDRATE = 4_000_000
PROTOCOL_VERSION = 2.0
ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64
ADDR_HARDWARE_ERROR_STATUS = 70
ADDR_GOAL_CURRENT = 102
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
POSITION_SCALE = 2.0 * 3.141592653589793 / 4096
CURRENT_SCALE_MA = 1.34
MOTOR_LABELS = {
    1: "index MCP forward",
    5: "middle MCP forward",
    6: "middle PIP",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pulse one LEAP motor at low current and restore its starting position."
    )
    parser.add_argument(
        "--motor",
        type=int,
        default=6,
        help="Dynamixel motor ID to test. Use 1, 5, or 6 here. Default: 6.",
    )
    parser.add_argument(
        "--delta-rad",
        type=float,
        default=0.12,
        help="Signed pulse displacement in motor radians. Default: 0.12.",
    )
    parser.add_argument(
        "--current-ma",
        type=float,
        default=100.0,
        help="Goal Current limit in mA. Keep this low for inspection. Default: 100.",
    )
    parser.add_argument(
        "--hold",
        type=float,
        default=1.0,
        help="Seconds to hold the pulse before returning. Default: 1.0.",
    )
    return parser.parse_args()


def candidate_ports() -> list[str]:
    ports: list[str] = []
    for pattern in ("/dev/cu.usbserial-*", "/dev/cu.usbmodem*", "/dev/ttyUSB0"):
        ports.extend(sorted(glob.glob(pattern)))
    return list(dict.fromkeys(ports))


def require_success(packet: PacketHandler, result: int, error: int, action: str) -> None:
    if result != COMM_SUCCESS:
        raise RuntimeError(f"{action}: {packet.getTxRxResult(result)}")
    if error:
        raise RuntimeError(f"{action}: {packet.getRxPacketError(error)}")


def to_signed_32(value: int) -> int:
    return value - (1 << 32) if value & (1 << 31) else value


def read1(packet: PacketHandler, port: PortHandler, motor_id: int, address: int) -> int:
    value, result, error = packet.read1ByteTxRx(port, motor_id, address)
    require_success(packet, result, error, f"read address {address}")
    return int(value)


def read4(packet: PacketHandler, port: PortHandler, motor_id: int, address: int) -> int:
    value, result, error = packet.read4ByteTxRx(port, motor_id, address)
    require_success(packet, result, error, f"read address {address}")
    return to_signed_32(int(value))


def write1(
    packet: PacketHandler, port: PortHandler, motor_id: int, address: int, value: int
) -> None:
    result, error = packet.write1ByteTxRx(port, motor_id, address, value)
    require_success(packet, result, error, f"write address {address}")


def write2(
    packet: PacketHandler, port: PortHandler, motor_id: int, address: int, value: int
) -> None:
    result, error = packet.write2ByteTxRx(port, motor_id, address, value)
    require_success(packet, result, error, f"write address {address}")


def write4(
    packet: PacketHandler, port: PortHandler, motor_id: int, address: int, value: int
) -> None:
    result, error = packet.write4ByteTxRx(port, motor_id, address, value & 0xFFFFFFFF)
    require_success(packet, result, error, f"write address {address}")


def main() -> None:
    args = parse_args()
    if args.motor not in MOTOR_LABELS:
        raise SystemExit("For this diagnostic, --motor must be one of: 1, 5, 6.")
    if args.delta_rad == 0:
        raise SystemExit("--delta-rad must be non-zero.")
    if not 0 < args.current_ma <= 350:
        raise SystemExit("For this inspection script, --current-ma must be in (0, 350].")

    ports = candidate_ports()
    if not ports:
        raise SystemExit("No LEAP Hand serial port found.")

    packet = PacketHandler(PROTOCOL_VERSION)
    port = None
    for port_name in ports:
        candidate = PortHandler(port_name)
        if candidate.openPort() and candidate.setBaudRate(BAUDRATE):
            port = candidate
            print(f"Connected on {port_name}")
            break
        candidate.closePort()
    if port is None:
        raise SystemExit("Could not open a LEAP Hand serial port at 4 Mbps.")

    torque_enabled = False
    try:
        mode = read1(packet, port, args.motor, ADDR_OPERATING_MODE)
        hardware_error = read1(packet, port, args.motor, ADDR_HARDWARE_ERROR_STATUS)
        if hardware_error:
            raise RuntimeError(
                f"ID {args.motor} has latched hardware error 0x{hardware_error:02x}; "
                "clear it before running a motion test."
            )
        if mode != 5:
            raise RuntimeError(
                f"ID {args.motor} operating mode is {mode}, expected 5 (current-based position mode)."
            )

        start_raw = read4(packet, port, args.motor, ADDR_PRESENT_POSITION)
        delta_raw = round(args.delta_rad / POSITION_SCALE)
        pulse_raw = start_raw + delta_raw
        current_raw = round(args.current_ma / CURRENT_SCALE_MA)
        print(
            f"ID {args.motor} normally drives {MOTOR_LABELS[args.motor]}. "
            f"Pulse: {start_raw} -> {pulse_raw} -> {start_raw}; "
            f"goal current: {current_raw} raw ({current_raw * CURRENT_SCALE_MA:.1f}mA)."
        )
        print("Watch the hand; press Ctrl+C to stop and restore the starting target.")

        write2(packet, port, args.motor, ADDR_GOAL_CURRENT, current_raw)
        write1(packet, port, args.motor, ADDR_TORQUE_ENABLE, 1)
        torque_enabled = True
        write4(packet, port, args.motor, ADDR_GOAL_POSITION, pulse_raw)
        time.sleep(max(args.hold, 0.0))
        pulse_position = read4(packet, port, args.motor, ADDR_PRESENT_POSITION)
        print(f"At pulse: present_raw={pulse_position}, delta={pulse_position - start_raw}")

        write4(packet, port, args.motor, ADDR_GOAL_POSITION, start_raw)
        time.sleep(max(args.hold, 0.0))
        restored_position = read4(packet, port, args.motor, ADDR_PRESENT_POSITION)
        print(f"After return: present_raw={restored_position}, delta={restored_position - start_raw}")
    except KeyboardInterrupt:
        print("Interrupted; restoring starting position if it was read.")
        if "start_raw" in locals():
            try:
                write4(packet, port, args.motor, ADDR_GOAL_POSITION, start_raw)
            except RuntimeError as exc:
                print(f"Could not restore Goal Position: {exc}", file=sys.stderr)
    finally:
        if torque_enabled:
            try:
                write1(packet, port, args.motor, ADDR_TORQUE_ENABLE, 0)
                write2(packet, port, args.motor, ADDR_GOAL_CURRENT, 0)
                print("Torque disabled.")
            except RuntimeError as exc:
                print(f"Could not disable torque: {exc}", file=sys.stderr)
        port.closePort()


if __name__ == "__main__":
    main()
