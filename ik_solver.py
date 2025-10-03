from trac_ik_python.trac_ik import IK
from geometry_msgs.msg import Pose

class IK_Solver:
    def __init__(self):
        self.ik = IK(
            base_link="torso_lift_link",
            tip_link="arm_tool_link",
            urdf_string=None,
            timeout=0.1,
            epsilon=1e-4
        )
        self.joint_names = self.ik.get_joint_names()

    def solve(self, pose: Pose):
        pos = pose.position
        ori = pose.orientation
        seed = [0.0] * len(self.joint_names)

        solution = self.ik.get_ik(
            seed,
            pos.x, pos.y, pos.z,
            ori.x, ori.y, ori.z, ori.w
        )
        return list(solution) if solution else None

