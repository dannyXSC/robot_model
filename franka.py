import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3
from urdf_parser_py.urdf import URDF
import trimesh

URDF_PATH = "./assets/urdf/franka.urdf"
PACKAGE_PATH = "./assets/"
DEFAULT_JOINT_ANGLES = np.array(
    [
        0.02597709,
        0.18678349,
        -0.02388557,
        -2.58533829,
        -0.01678507,
        2.97463095,
        0.78241131,
        0.020833,
    ]
)


class Franka_TF(rtb.models.Panda):
    def __init__(self):
        super().__init__()
        self.hand_tf = self.grippers[0].links[0].A()
        self.left_R = self.grippers[0].links[1].A()
        self.right_R = self.grippers[0].links[2].A()

    def fkine_all(self, q):
        joint_angles = q[:-1]
        gripper_q = q[-1] / 2
        tf_list = super().fkine_all(joint_angles)
        tf_body_list = tf_list[1:9]
        link_8 = tf_list[9]
        tf_hand = link_8 @ self.grippers[0].links[0].A()
        tf_leftfinger = tf_hand @ SE3(0, gripper_q, 0) @ self.left_R
        tf_rightfinger = (
            tf_hand @ SE3.Rz(3.141592653589793) @ SE3(0, gripper_q, 0) @ self.right_R
        )
        result = SE3(
            [T for T in tf_body_list] + [tf_hand, tf_leftfinger, tf_rightfinger]
        )
        return result


class Franka_Model:
    def __init__(self):
        self.bbox = []
        self.link_num = 0
        self.robot_tf = None
        self.joint_angles = []

        self._load_urdf()
        self.reset()

    def _load_urdf(self):
        with open(URDF_PATH, "r") as f:
            urdf_str = f.read()
        urdf_str = urdf_str.replace("package://", PACKAGE_PATH)
        self.robot_urdf = URDF.from_xml_string(urdf_str)
        for link in self.robot_urdf.links:
            for visual in link.visuals:
                mesh_file = link.visual.geometry.filename
                if mesh_file is None:
                    continue
                mesh_trimesh = trimesh.load(mesh_file, force="mesh")
                if isinstance(mesh_trimesh, trimesh.Scene):
                    mesh_trimesh = trimesh.util.concatenate(mesh_trimesh.dump())
                self.bbox.append(mesh_trimesh.bounds)
        self.link_num = len(self.bbox)

    def reset(self, joint_angles=DEFAULT_JOINT_ANGLES):
        self.robot_tf = Franka_TF()
        self.joint_angles = joint_angles
        self.current_tf = self.robot_tf.fkine_all(self.joint_angles)

    def point_to_link(self, points):
        bbox_array = np.asarray(self.bbox)  # shape: (link_num, 2, 3)
        # 检查每个点所在的link索引
        link_indices = np.array([-1 for _ in range(len(points))])
        points_hom = np.hstack([points, np.ones((points.shape[0], 1))])

        for i in range(self.link_num):
            tf = self.current_tf[i]
            local_points = (tf.inv().A @ points_hom.T).T[:, :3]

            bbox_min = bbox_array[i, 0, :3]  # 第i个link的最小点
            bbox_max = bbox_array[i, 1, :3]  # 第i个link的最大点
            condition_mask = (local_points >= bbox_min) & (local_points <= bbox_max)
            in_link_i = np.all(condition_mask, axis=1)
            # 将满足条件的点的索引设置为当前link索引
            link_indices[in_link_i] = i

        return link_indices


if __name__ == "__main__":
    franka = Franka_Model()

    import pdb

    pdb.set_trace()
    input()
