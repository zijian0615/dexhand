import numpy as np
import os
import time

try:
    from hand_tracking_sdk.frame import HandFrame
except ImportError:
    HandFrame = None

from quest_hand_pos import frame_matches_hand, quest_frame_to_hand_pos

try:
    from avp_stream import VisionProStreamer
except ImportError:
    VisionProStreamer = None


def _import_pybullet():
    try:
        import pybullet as p
    except ImportError as exc:
        raise ImportError(
            "PyBullet is required for IK simulation but is not installed. "
            "On macOS it often fails to build from source. "
            "For now, verify HTS with:\n"
            "  uv run python Bidex_VisionPro_Teleop/quest_stream_test.py --port 8000"
        ) from exc
    return p

'''
This is based off of https://github.com/Improbable-AI/VisionProTeleop by Younghyo Park et. al. which streams the AVP data.
We then runs inverse kinematics and output LEAP Hand joint angles.
Feed these joint angles into your preferred LEAP Hand API, Python, ROS1/ROS2 etc.

See https://github.com/leap-hand/Bidex_VisionPro_Teleop for further details.

Note how the fingertip positions are matching, but the joint angles between the two hands are not due to the IK solution.  :) 
This is not as accurate as Manus gloves but easier to use.
Inspired by Dexcap https://dex-cap.github.io/ by Wang et. al. and Robotic Telekinesis by Shaw et. al.
'''

#Set this to your IP address of you AVP
AVP_IP = "172.26.16.138"

class Leapv1PybulletIKPython():
    def __init__(self, is_left=True, use_avp=False, avp_ip=AVP_IP, gui=True):
        p = _import_pybullet()
        self.p = p
        self.vps = None
        if use_avp:
            if VisionProStreamer is None:
                raise ImportError(
                    "Vision Pro teleop requires avp_stream. Install with: pip install avp_stream"
                )
            self.vps = VisionProStreamer(ip=avp_ip, record=True)
        # start pybullet
        #clid = p.connect(p.SHARED_MEMORY)
        # Use DIRECT (headless) on a Jetson with no display, GUI otherwise.
        p.connect(p.GUI if gui else p.DIRECT)
        # load right leap hand           
        path_src = os.path.abspath(__file__)
        path_src = os.path.dirname(path_src)
        self.is_left = is_left
        self.glove_to_leap_mapping_scale = 1.6
        self.leapEndEffectorIndex = [3, 4, 8, 9, 13, 14, 18, 19]
        if self.is_left:
            path_src = os.path.join(path_src, "leap_hand_mesh_left/robot_pybullet.urdf")
            ##You may have to set this path for your setup on ROS2
            self.LeapId = p.loadURDF(
                path_src,
                [0.31, 0.01, 0.06],
                p.getQuaternionFromEuler([1.57, 0, 0]),   ##this is the correct frame and rotation for left hand.....use left hand data instead
                useFixedBase = True
            )
        else:
            path_src = os.path.join(path_src, "leap_hand_mesh_right/robot_pybullet.urdf")
            ##You may have to set this path for your setup on ROS2
            self.LeapId = p.loadURDF(
                path_src,
                [-0.22, 0.01, 0.03],
                p.getQuaternionFromEuler([1.57, 0, 3.14]),
                useFixedBase = True
            )

        self.numJoints = p.getNumJoints(self.LeapId)
        p.setGravity(0, 0, 0)
        useRealTimeSimulation = 0
        p.setRealTimeSimulation(useRealTimeSimulation)
        self.create_target_vis()
            
    def create_target_vis(self):
        p = self.p
        # load balls
        small_ball_radius = 0.01
        small_ball_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=small_ball_radius)
        ball_radius = 0.01
        ball_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=ball_radius)
        baseMass = 0.001
        basePosition = [0.25, 0.25, 0]
        
        self.ballMbt = []
        for i in range(0,4):
            self.ballMbt.append(p.createMultiBody(baseMass=baseMass, baseCollisionShapeIndex=ball_shape, basePosition=basePosition)) # for base and finger tip joints    
            no_collision_group = 0
            no_collision_mask = 0
            p.setCollisionFilterGroupMask(self.ballMbt[i], -1, no_collision_group, no_collision_mask)
        p.changeVisualShape(self.ballMbt[0], -1, rgbaColor=[1, 0, 0, 1]) 
        p.changeVisualShape(self.ballMbt[1], -1, rgbaColor=[0, 1, 0, 1]) 
        p.changeVisualShape(self.ballMbt[2], -1, rgbaColor=[0, 0, 1, 1])  
        p.changeVisualShape(self.ballMbt[3], -1, rgbaColor=[1, 1, 1, 1])
        
    def update_target_vis(self, hand_pos):
        p = self.p
        _, current_orientation = p.getBasePositionAndOrientation( self.ballMbt[0])
        p.resetBasePositionAndOrientation(self.ballMbt[0], hand_pos[3], current_orientation)
        _, current_orientation = p.getBasePositionAndOrientation(self.ballMbt[1])
        p.resetBasePositionAndOrientation(self.ballMbt[1], hand_pos[2], current_orientation)
        _, current_orientation = p.getBasePositionAndOrientation(self.ballMbt[2])
        p.resetBasePositionAndOrientation(self.ballMbt[2], hand_pos[7], current_orientation)
        _, current_orientation = p.getBasePositionAndOrientation(self.ballMbt[3])
        p.resetBasePositionAndOrientation(self.ballMbt[3], hand_pos[1], current_orientation)
        
    def get_glove_data(self, hand_pos):
        #gets the data converts it and then computes IK and visualizes
        # hand_pos[2][0] = hand_pos[2][0] - 0.02  this isn't great because they won't oppose properly
        # hand_pos[3][0] = hand_pos[3][0] - 0.02    
        # hand_pos[6][0] = hand_pos[6][0] + 0.02
        # hand_pos[7][0] = hand_pos[7][0] + 0.02
        #hand_pos[2][1] = hand_pos[2][1] + 0.002
        hand_pos[4][1] = hand_pos[4][1] + 0.002
        hand_pos[6][1] = hand_pos[6][1] + 0.002
        output = self.compute_IK(hand_pos)
        self.update_target_vis(hand_pos)
        return output
    

    def get_quest_data(self, frame):
        """Convert one Quest HTS HandFrame into 16 LEAP joint angles."""
        if HandFrame is None:
            raise ImportError(
                "Quest teleop requires hand-tracking-sdk. Install with: pip install hand-tracking-sdk"
            )
        if not isinstance(frame, HandFrame):
            raise TypeError(f"Expected HandFrame, got {type(frame)!r}")
        if not frame_matches_hand(frame, self.is_left):
            return None

        hand_pos = quest_frame_to_hand_pos(frame)
        return self.get_glove_data(hand_pos)

    def get_avp_data(self):
        #gets the data converts it and then computes IK and visualizes
        if self.vps is None:
            raise RuntimeError(
                "Vision Pro streamer is not initialized. Pass use_avp=True to __init__()."
            )
        r = self.vps.latest              
        if self.is_left:
            hand_pose = np.asarray(r['left_fingers']).astype(float)  
        else:
            hand_pose = np.asarray(r['right_fingers']).astype(float) 
        indices = [3,4,8,9,13,14,18,19,23,24]
        hand_pos = hand_pose[indices, :3, 3]
        for i in range(0,10):
            hand_pos[i][0] = hand_pos[i][0] * 1.35 * 1.5
            hand_pos[i][1] = hand_pos[i][1] * 1.5
            hand_pos[i][2] = hand_pos[i][2] * 1.5
        output = self.compute_IK(hand_pos)
        self.update_target_vis(hand_pos)
        return output
        
    def compute_IK(self, hand_pos):
        p = self.p
        p.stepSimulation()     

        rightHandIndex_middle_pos = hand_pos[2]
        rightHandIndex_pos = hand_pos[3]
        
        rightHandMiddle_middle_pos = hand_pos[4]
        rightHandMiddle_pos = hand_pos[5]
        
        rightHandRing_middle_pos = hand_pos[6]
        rightHandRing_pos = hand_pos[7]
        
        rightHandThumb_middle_pos = hand_pos[0]
        rightHandThumb_pos = hand_pos[1]
        
        leapEndEffectorPos = [
            rightHandIndex_middle_pos,
            rightHandIndex_pos,
            rightHandMiddle_middle_pos,
            rightHandMiddle_pos,
            rightHandRing_middle_pos,
            rightHandRing_pos,
            rightHandThumb_middle_pos,
            rightHandThumb_pos
        ]

        jointPoses = p.calculateInverseKinematics2(
            self.LeapId,
            self.leapEndEffectorIndex,
            leapEndEffectorPos,
            solver=p.IK_DLS,
            maxNumIterations=50,
            residualThreshold=0.0001,
        )
        
        combined_jointPoses = (jointPoses[0:4] + (0.0,) + jointPoses[4:8] + (0.0,) + jointPoses[8:12] + (0.0,) + jointPoses[12:16] + (0.0,))
        combined_jointPoses = list(combined_jointPoses)

        # update the hand joints
        for i in range(20):
            p.setJointMotorControl2(
                bodyIndex=self.LeapId,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=combined_jointPoses[i],
                targetVelocity=0,
                force=500,
                positionGain=0.3,
                velocityGain=1,
            )

        # map results to real robot
        real_robot_hand_q = np.array([float(0.0) for _ in range(16)])
        #real_left_robot_hand_q = np.array([0.0 for _ in range(16)])

        real_robot_hand_q[0:4] = jointPoses[0:4]
        real_robot_hand_q[4:8] = jointPoses[4:8]
        real_robot_hand_q[8:12] = jointPoses[8:12]
        real_robot_hand_q[12:16] = jointPoses[12:16]
        real_robot_hand_q[0:2] = real_robot_hand_q[0:2][::-1]
        real_robot_hand_q[4:6] = real_robot_hand_q[4:6][::-1]
        real_robot_hand_q[8:10] = real_robot_hand_q[8:10][::-1]
        return [float(i) for i in real_robot_hand_q]

if __name__ == "__main__":
    pbik = Leapv1PybulletIKPython()
    while True:
        output_joints = pbik.get_avp_data()
        print(output_joints)  ##This is the 16 dimensional LEAP Hand vector  Feed this into your python or ROS1/ROS2 code.  Ordering is the same!
        time.sleep(0.03)