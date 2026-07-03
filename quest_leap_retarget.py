"""Retarget Quest HTS HandFrame to LEAP Hand 16-DoF allegro joint angles."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from hand_tracking_sdk.convert import convert_hand_frame_unity_left_to_right
from hand_tracking_sdk.frame import HandFrame
from hand_tracking_sdk.models import JointName
from hand_tracking_sdk.teleop import _FINGER_CHAINS

from quest_hand_pos import _landmark_in_wrist_frame, _quat_to_rotation_matrix


@dataclass
class LeapRetargetConfig:
    """Tunable mapping from HTS landmarks to LEAP allegro joint angles."""

    spread_scale: float = 0.8
    max_spread_rad: float = 0.45
    enable_spread: bool = False
    mcp_forward_scale: float = 1.2
    pip_scale: float = 1.3
    dip_scale: float = 1.1
    thumb_scale: float = 1.2
    mcp_closure_weight: float = 0.40
    pip_closure_weight: float = 0.85
    dip_closure_weight: float = 0.55
    fist_sync: float = 0.55
    fist_follow: float = 1.0
    fist_threshold: float = 0.025
    mirror_fist_joints: bool = True
    mirror_blend: float = 0.98
    finger_gain: dict[str, float] = field(
        default_factory=lambda: {
            "index": 1.05,
            "middle": 1.08,
            "ring": 1.00,
            "thumb": 1.10,
        }
    )
    smoothing: float = 0.15
    max_joint_rad: float = 1.2
    max_curl_delta: float = 0.55
    calibrate_frames: int = 45


_FINGER_BASES = {
    "index": 0,
    "middle": 4,
    "ring": 8,
}
_FINGER_ORDER = ("index", "middle", "ring", "thumb")
_SPREAD_JOINTS = {
    "index": JointName.INDEX_PROXIMAL,
    "middle": JointName.MIDDLE_PROXIMAL,
    "ring": JointName.RING_PROXIMAL,
    "thumb": JointName.THUMB_METACARPAL,
}


@dataclass
class _OpenHandBaseline:
    curl_sum: dict[str, np.ndarray] = field(default_factory=dict)
    spread_sum: dict[str, float] = field(default_factory=dict)
    count: int = 0
    ready: bool = False

    def reset(self) -> None:
        self.curl_sum.clear()
        self.spread_sum.clear()
        self.count = 0
        self.ready = False

    def add(self, curls: dict[str, tuple[float, ...]], spreads: dict[str, float]) -> None:
        for finger, angles in curls.items():
            arr = np.asarray(angles, dtype=float)
            if finger not in self.curl_sum:
                self.curl_sum[finger] = arr.copy()
            else:
                self.curl_sum[finger] = self.curl_sum[finger] + arr
        for finger, value in spreads.items():
            self.spread_sum[finger] = self.spread_sum.get(finger, 0.0) + value
        self.count += 1

    def finalize(self) -> None:
        if self.count <= 0:
            return
        for finger in list(self.curl_sum):
            self.curl_sum[finger] /= self.count
        for finger in list(self.spread_sum):
            self.spread_sum[finger] /= self.count
        self.ready = True


class QuestLeapRetargeter:
    """Map HTS frames to 16-D allegro joint commands for LEAP Hand."""

    def __init__(self, config: LeapRetargetConfig | None = None):
        self.config = config or LeapRetargetConfig()
        self._prev: np.ndarray | None = None
        self._baseline = _OpenHandBaseline()

    @property
    def calibrated(self) -> bool:
        return self._baseline.ready

    @property
    def calibrate_progress(self) -> float:
        if self._baseline.ready:
            return 1.0
        if self.config.calibrate_frames <= 0:
            return 1.0
        return min(1.0, self._baseline.count / self.config.calibrate_frames)

    def reset(self) -> None:
        self._prev = None
        self._baseline.reset()

    def retarget(self, frame: HandFrame) -> np.ndarray:
        curls, spreads = self._measure(frame)

        if not self._baseline.ready:
            self._baseline.add(curls, spreads)
            if self._baseline.count >= self.config.calibrate_frames:
                self._baseline.finalize()
            return np.zeros(16, dtype=float)

        joints = self._map_to_leap(curls, spreads)
        joints = np.clip(joints, 0.0, self.config.max_joint_rad)

        alpha = float(np.clip(self.config.smoothing, 0.0, 1.0))
        if alpha <= 0.0:
            self._prev = joints.copy()
        elif self._prev is None:
            self._prev = joints.copy()
        else:
            joints = (1.0 - alpha) * self._prev + alpha * joints
            self._prev = joints.copy()
        return joints

    def _measure(self, frame: HandFrame) -> tuple[dict[str, tuple[float, ...]], dict[str, float]]:
        frame = convert_hand_frame_unity_left_to_right(frame)
        wrist = frame.wrist
        wrist_pos = np.array([wrist.x, wrist.y, wrist.z], dtype=float)
        wrist_rot = _quat_to_rotation_matrix(wrist.qx, wrist.qy, wrist.qz, wrist.qw)

        curls = _finger_curl_angles_wrist(frame, wrist_pos, wrist_rot, _FINGER_ORDER)
        spreads = {
            finger: _knuckle_spread(frame, joint, wrist_pos, wrist_rot)
            for finger, joint in _SPREAD_JOINTS.items()
        }
        return curls, spreads

    def _map_to_leap(
        self,
        curls: dict[str, tuple[float, ...]],
        spreads: dict[str, float],
    ) -> np.ndarray:
        cfg = self.config
        joints = np.zeros(16, dtype=float)

        finger_deltas: dict[str, tuple[float, float, float]] = {}
        finger_closure: dict[str, float] = {}
        for finger, base in _FINGER_BASES.items():
            c0, c1, c2 = curls[finger]
            b0, b1, b2 = self._baseline.curl_sum[finger]
            deltas = _curl_deltas(c0, c1, c2, b0, b1, b2, cfg.max_curl_delta)
            finger_deltas[finger] = deltas
            finger_closure[finger] = max(deltas)

        group_closure = max(finger_closure.values()) if finger_closure else 0.0
        sync = float(np.clip(cfg.fist_sync, 0.0, 1.0))
        follow = float(np.clip(cfg.fist_follow, 0.0, 1.0))

        for finger, base in _FINGER_BASES.items():
            d0, d1, d2 = finger_deltas[finger]
            per_finger = finger_closure[finger]
            synced = (1.0 - sync) * per_finger + sync * group_closure
            # When one finger tracks a fist, weak/occluded fingers still follow it.
            blended = max(synced, group_closure * follow)
            gain = cfg.finger_gain.get(finger, 1.0)
            closure = min(blended * gain, cfg.max_curl_delta)
            if cfg.enable_spread:
                spread_delta = (spreads[finger] - self._baseline.spread_sum[finger]) * cfg.spread_scale
                joints[base + 0] = _spread_to_allegro(spread_delta, cfg.max_spread_rad)
            joints[base + 1] = max(d0, closure * cfg.mcp_closure_weight) * cfg.mcp_forward_scale
            joints[base + 2] = max(d1, closure * cfg.pip_closure_weight) * cfg.pip_scale
            joints[base + 3] = max(d2, closure * cfg.dip_closure_weight) * cfg.dip_scale

        t0, t1, t2 = curls["thumb"]
        tb0, tb1, tb2 = self._baseline.curl_sum["thumb"]
        td0, td1, td2 = _curl_deltas(t0, t1, t2, tb0, tb1, tb2, cfg.max_curl_delta)
        thumb_per = max(td0, td1, td2)
        thumb_synced = (1.0 - sync) * thumb_per + sync * group_closure
        thumb_blended = max(thumb_synced, group_closure * follow * 0.90, thumb_per)
        t_closure = min(
            thumb_blended * cfg.finger_gain.get("thumb", 1.0),
            cfg.max_curl_delta,
        )
        if cfg.enable_spread:
            spread_delta = (spreads["thumb"] - self._baseline.spread_sum["thumb"]) * cfg.spread_scale
            joints[12] = _spread_to_allegro(spread_delta, cfg.max_spread_rad)
        joints[13] = max(td0, t_closure * 0.55) * cfg.thumb_scale
        joints[14] = max(td1, t_closure * 0.75) * cfg.thumb_scale
        joints[15] = max(td2, t_closure * 0.45) * cfg.thumb_scale

        if cfg.mirror_fist_joints and group_closure >= cfg.fist_threshold:
            joints = _mirror_fist_joints(joints, finger_closure, cfg)

        return joints

    def debug_curl_deltas(self, frame: HandFrame) -> dict[str, list[float]]:
        """Debug helper: curl angles minus open-hand baseline (positive = closing)."""
        curls, _ = self._measure(frame)
        if not self._baseline.ready:
            return {finger: [0.0, 0.0, 0.0] for finger in _FINGER_ORDER}

        out: dict[str, list[float]] = {}
        per_finger: dict[str, float] = {}
        for finger in _FINGER_ORDER:
            c = curls[finger]
            b = self._baseline.curl_sum[finger]
            deltas = [float(max(0.0, c[i] - b[i])) for i in range(3)]
            out[finger] = deltas
            if finger != "thumb":
                per_finger[finger] = max(deltas)
        group = max(per_finger.values()) if per_finger else 0.0
        sync = float(np.clip(self.config.fist_sync, 0.0, 1.0))
        follow = float(np.clip(self.config.fist_follow, 0.0, 1.0))
        out["_group_closure"] = [group]
        for finger in ("index", "middle", "ring"):
            gain = self.config.finger_gain.get(finger, 1.0)
            per = per_finger.get(finger, 0.0)
            synced = (1.0 - sync) * per + sync * group
            blended = max(synced, group * follow)
            out[f"_{finger}_eff"] = [blended * gain]
        if self.config.mirror_fist_joints and group >= self.config.fist_threshold:
            leader = max(per_finger, key=per_finger.get)
            out["_leader"] = [leader]
        return out

    def baseline_summary(self) -> dict[str, list[float]]:
        if not self._baseline.ready:
            return {}
        return {finger: self._baseline.curl_sum[finger].tolist() for finger in _FINGER_ORDER}


def _spread_to_allegro(delta: float, max_abs: float) -> float:
    return float(np.clip(delta, -max_abs, max_abs))


def _mirror_fist_joints(
    joints: np.ndarray,
    finger_closure: dict[str, float],
    cfg: LeapRetargetConfig,
) -> np.ndarray:
    """Copy the best-tracked finger's curl joints onto the other fingers."""
    leader = max(finger_closure, key=finger_closure.get)
    if finger_closure[leader] < cfg.fist_threshold:
        return joints

    src_base = _FINGER_BASES[leader]
    src = joints[src_base + 1 : src_base + 4]
    leader_gain = cfg.finger_gain.get(leader, 1.0)
    blend = float(np.clip(cfg.mirror_blend, 0.0, 1.0))

    for finger, base in _FINGER_BASES.items():
        if finger == leader:
            continue
        rel_gain = cfg.finger_gain.get(finger, 1.0) / max(leader_gain, 1e-6)
        for i in range(3):
            mirrored = src[i] * rel_gain * blend
            joints[base + 1 + i] = max(joints[base + 1 + i], mirrored)

    thumb_gain = cfg.finger_gain.get("thumb", 1.0) / max(leader_gain, 1e-6)
    joints[14] = max(joints[14], src[1] * thumb_gain * blend * 0.85)
    joints[15] = max(joints[15], src[2] * thumb_gain * blend * 0.70)
    return joints


def _curl_deltas(
    c0: float,
    c1: float,
    c2: float,
    b0: float,
    b1: float,
    b2: float,
    max_delta: float | None = None,
) -> tuple[float, float, float]:
    deltas = (
        max(0.0, c0 - b0),
        max(0.0, c1 - b1),
        max(0.0, c2 - b2),
    )
    if max_delta is None:
        return deltas
    cap = max(0.0, max_delta)
    return tuple(min(d, cap) for d in deltas)


def _joint_in_wrist_frame(
    frame: HandFrame,
    joint: JointName,
    wrist_pos: np.ndarray,
    wrist_rot: np.ndarray,
) -> tuple[float, float, float]:
    rel = _landmark_in_wrist_frame(
        wrist_pos[0],
        wrist_pos[1],
        wrist_pos[2],
        wrist_rot,
        frame.get_joint(joint),
    )
    return float(rel[0]), float(rel[1]), float(rel[2])


def _angle_between(
    u: tuple[float, float, float],
    v: tuple[float, float, float],
) -> float:
    dot = u[0] * v[0] + u[1] * v[1] + u[2] * v[2]
    len_u = math.sqrt(u[0] ** 2 + u[1] ** 2 + u[2] ** 2)
    len_v = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    denom = len_u * len_v
    if denom == 0.0:
        return 0.0
    return math.acos(max(-1.0, min(1.0, dot / denom)))


def _vec_sub(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _finger_curl_angles_wrist(
    frame: HandFrame,
    wrist_pos: np.ndarray,
    wrist_rot: np.ndarray,
    fingers: tuple[str, ...],
) -> dict[str, tuple[float, ...]]:
    """Inter-joint bend angles computed in the wrist frame (rotation invariant)."""
    result: dict[str, tuple[float, ...]] = {}
    for name in fingers:
        chain = _FINGER_CHAINS[name]
        positions = [_joint_in_wrist_frame(frame, joint, wrist_pos, wrist_rot) for joint in chain]
        angles: list[float] = []
        for i in range(1, len(positions) - 1):
            incoming = _vec_sub(positions[i], positions[i - 1])
            outgoing = _vec_sub(positions[i + 1], positions[i])
            angles.append(_angle_between(incoming, outgoing))
        result[name] = tuple(angles)
    return result


def _knuckle_spread(
    frame: HandFrame,
    joint: JointName,
    wrist_pos: np.ndarray,
    wrist_rot: np.ndarray,
) -> float:
    rel = _landmark_in_wrist_frame(
        wrist_pos[0],
        wrist_pos[1],
        wrist_pos[2],
        wrist_rot,
        frame.get_joint(joint),
    )
    return float(rel[0])
