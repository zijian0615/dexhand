"""Convert Quest HTS HandFrame data into Bidex LEAP IK hand_pos format."""

from __future__ import annotations

import numpy as np

try:
    from hand_tracking_sdk.convert import convert_hand_frame_unity_left_to_right
    from hand_tracking_sdk.frame import HandFrame
    from hand_tracking_sdk.models import HandSide, JointName
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "Quest teleop requires hand-tracking-sdk. Install with: pip install hand-tracking-sdk"
    ) from exc

# Matches get_avp_data() ordering: thumb, index, middle, ring, little (middle+tip each).
# compute_IK() only reads indices 0-7; 8-9 are kept for parity with AVP scaling.
QUEST_IK_JOINTS: tuple[tuple[JointName, JointName], ...] = (
    (JointName.THUMB_DISTAL, JointName.THUMB_TIP),
    (JointName.INDEX_INTERMEDIATE, JointName.INDEX_TIP),
    (JointName.MIDDLE_INTERMEDIATE, JointName.MIDDLE_TIP),
    (JointName.RING_INTERMEDIATE, JointName.RING_TIP),
    (JointName.LITTLE_INTERMEDIATE, JointName.LITTLE_TIP),
)

# Same empirical scaling used in avp_leap.get_avp_data().
DEFAULT_X_SCALE = 1.35 * 1.5
DEFAULT_YZ_SCALE = 1.5


def _quat_to_rotation_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=float,
    )


def _landmark_in_wrist_frame(
    wrist_x: float,
    wrist_y: float,
    wrist_z: float,
    wrist_rot: np.ndarray,
    landmark_xyz: tuple[float, float, float],
) -> np.ndarray:
    """Express one world-space landmark in the wrist frame."""
    relative = np.asarray(landmark_xyz, dtype=float) - np.array([wrist_x, wrist_y, wrist_z])
    return wrist_rot.T @ relative


def quest_frame_to_hand_pos(
    frame: HandFrame,
    *,
    x_scale: float = DEFAULT_X_SCALE,
    yz_scale: float = DEFAULT_YZ_SCALE,
) -> np.ndarray:
    """Map one HTS HandFrame to the 10x3 array expected by Leap IK."""
    frame = convert_hand_frame_unity_left_to_right(frame)
    wrist = frame.wrist
    wrist_rot = _quat_to_rotation_matrix(wrist.qx, wrist.qy, wrist.qz, wrist.qw)

    hand_pos = np.zeros((10, 3), dtype=float)
    for finger_idx, (middle_joint, tip_joint) in enumerate(QUEST_IK_JOINTS):
        base = finger_idx * 2
        middle_xyz = frame.get_joint(middle_joint)
        tip_xyz = frame.get_joint(tip_joint)
        hand_pos[base] = _landmark_in_wrist_frame(
            wrist.x, wrist.y, wrist.z, wrist_rot, middle_xyz
        )
        hand_pos[base + 1] = _landmark_in_wrist_frame(
            wrist.x, wrist.y, wrist.z, wrist_rot, tip_xyz
        )

    hand_pos[:, 0] *= x_scale
    hand_pos[:, 1] *= yz_scale
    hand_pos[:, 2] *= yz_scale
    return hand_pos


def frame_matches_hand(frame: HandFrame, is_left: bool) -> bool:
    """Return True when HTS frame side matches the loaded LEAP URDF side."""
    expected = HandSide.LEFT if is_left else HandSide.RIGHT
    return frame.side == expected
