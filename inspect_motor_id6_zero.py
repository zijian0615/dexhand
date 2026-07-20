"""Read-only mechanical-reference inspection for a LEAP Hand motor.

This does not move the motor or change any registers. It reports the Dynamixel
Homing Offset register and the current encoder position. A mechanical zero is
not stored directly by the motor: place the physical joint at a known reference
pose first, then record Present Position as that pose's encoder coordinate.
"""

from __future__ import annotations

import argparse
import glob

from dynamixel_sdk import COMM_SUCCESS, PacketHandler, PortHandler

BAUDRATE = 4_000_000
POSITION_SCALE = 2.0 * 3.141592653589793 / 4096
ENCODER_COUNTS_PER_REVOLUTION = 4096
NOMINAL_OPEN_RAW = 2048

REGISTERS = {
    "operating_mode": (11, 1),
    "homing_offset": (20, 4),
    "torque_enable": (64, 1),
    "hardware_error": (70, 1),
    "goal_position": (116, 4),
    "present_position": (132, 4),
    "max_position_limit": (48, 4),
    "min_position_limit": (52, 4),
}


def to_signed(value: int, size: int) -> int:
    bits = size * 8
    return value - (1 << bits) if value & (1 << (bits - 1)) else value


def shortest_encoder_delta(current_raw: int, target_raw: int) -> int:
    """Return the shortest signed move from current to target in one revolution."""
    return (target_raw - current_raw + ENCODER_COUNTS_PER_REVOLUTION // 2) % (
        ENCODER_COUNTS_PER_REVOLUTION
    ) - ENCODER_COUNTS_PER_REVOLUTION // 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only LEAP/Dynamixel mechanical-reference inspection."
    )
    parser.add_argument(
        "--motor",
        type=int,
        default=6,
        help="Dynamixel motor ID to inspect. Default: 6.",
    )
    return parser.parse_args()


def ports() -> list[str]:
    found: list[str] = []
    for pattern in ("/dev/cu.usbserial-*", "/dev/cu.usbmodem*", "/dev/ttyUSB0"):
        found.extend(sorted(glob.glob(pattern)))
    return list(dict.fromkeys(found))


def read(
    packet: PacketHandler, port: PortHandler, motor_id: int, address: int, size: int
) -> int:
    method = {1: packet.read1ByteTxRx, 4: packet.read4ByteTxRx}[size]
    value, result, error = method(port, motor_id, address)
    if result != COMM_SUCCESS:
        raise RuntimeError(packet.getTxRxResult(result))
    if error:
        raise RuntimeError(packet.getRxPacketError(error))
    return int(value)


def main() -> None:
    args = parse_args()
    if not 0 <= args.motor <= 252:
        raise SystemExit("--motor must be a valid Dynamixel ID (0-252).")
    packet = PacketHandler(2.0)
    for name in ports():
        port = PortHandler(name)
        if not port.openPort() or not port.setBaudRate(BAUDRATE):
            port.closePort()
            continue
        try:
            values = {
                key: read(packet, port, args.motor, *spec)
                for key, spec in REGISTERS.items()
            }
        finally:
            port.closePort()
        for key in ("homing_offset", "goal_position", "present_position", "min_position_limit", "max_position_limit"):
            values[key] = to_signed(values[key], 4)

        print(f"Connected on {name}; motor ID {args.motor}")
        for key, value in values.items():
            print(f"{key}={value}")
        present = values["present_position"]
        present_modulo = present % ENCODER_COUNTS_PER_REVOLUTION
        nominal_open_delta = shortest_encoder_delta(present_modulo, NOMINAL_OPEN_RAW)
        print(f"present_position_rad={present * POSITION_SCALE:.6f}")
        print(f"present_position_mod_4096={present_modulo}")
        print("\nInterpretation:")
        print("- homing_offset is EEPROM configuration, not the physical joint zero itself.")
        print("- This is only a reading of the current pose; it establishes no zero unless the joint was deliberately placed at a known reference pose.")
        print(f"- present_position_mod_4096 normalizes extended position to one encoder turn ({ENCODER_COUNTS_PER_REVOLUTION} counts).")
        print(f"- LEAP API nominal open coordinate is near raw {NOMINAL_OPEN_RAW}; the shortest encoder difference from this current pose is {nominal_open_delta:+d} counts.")
        print("- Do not write Homing Offset from this difference. Confirm the mechanical reference and horn/gear assembly first.")
        return
    raise SystemExit("Could not open a LEAP Hand serial port at 4 Mbps.")


if __name__ == "__main__":
    main()
