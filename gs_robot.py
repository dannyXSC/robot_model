import numpy as np


from .gs_model import GS_MODEL
from .utils import point_to_hom, hom_to_point, to_hom, from_hom


class GS_ROBOT:

    def __init__(self, gs_model: GS_MODEL, robot, tf_gs_urdf):
        self.base_model = gs_model

        self.model = gs_model
        self.pre_model = self.model
        self.urdf_robot = robot
        self.tf_gs_urdf = tf_gs_urdf

        self.link_num = self.urdf_robot.link_num
        self.link_to_id = [[] for _ in range(self.link_num)]

        self.tailor()
        self.id_to_link = [-1 for _ in range(self.model.N)]

        self.gs_to_link()
        self.reset()

    def reset(self, q=None):
        """
        edit color
        divide gs into links
        edit position and rotation for each link
        """
        self.urdf_robot.reset(q)

    def gs_to_link(self):
        """
        divide gs into links
        """
        means = self.model.means
        urdf_pcd = (self.tf_gs_urdf @ point_to_hom(means).T).T
        urdf_pcd = hom_to_point(urdf_pcd)

        self.id_to_link = self.urdf_robot.point_to_link(urdf_pcd)

        for i in range(self.model.N):
            link_id = self.id_to_link[i]
            if link_id != -1:
                self.link_to_id[link_id].append(i)

    def tailor(self):
        means = self.model.means
        urdf_pcd = (self.tf_gs_urdf @ point_to_hom(means).T).T
        urdf_pcd = hom_to_point(urdf_pcd)
        self.id_to_link = self.urdf_robot.point_to_link(urdf_pcd)

        valid_idx = [i for i in range(self.model.N) if self.id_to_link[i] != -1]
        self.model = self.base_model[valid_idx]
        self.pre_model = self.model

    def get_model(self, q=None):
        if q is None:
            return self.pre_model
        link_tfs = self.urdf_robot.get_tf(q)

        means = self.model.means
        quats = self.model.quats
        scales = self.model.scales
        opacities = self.model.opacities
        colors = self.model.colors

        new_means = []
        new_quats = []
        new_scales = []
        new_opacities = []
        new_colors = []

        for i in range(self.link_num):
            gs_in_cur_link = self.link_to_id[i]
            tf = link_tfs[i]
            for idx in gs_in_cur_link:
                mean = means[idx]
                quat = quats[idx]
                scale = scales[idx]
                opacity = opacities[idx]
                color = colors[idx]

                hom_matrix = to_hom(mean, quat)
                new_pose = (
                    np.linalg.inv(self.tf_gs_urdf) @ tf.A @ self.tf_gs_urdf @ hom_matrix
                )
                new_mean, new_quat = from_hom(new_pose)
                new_scale = scale
                new_opacity = opacity
                # new_color = np.zeros_like(color)
                new_color = color

                new_means.append(new_mean)
                new_quats.append(new_quat)
                new_scales.append(new_scale)
                new_opacities.append(new_opacity)
                new_colors.append(new_color)

        new_means = np.array(new_means, dtype=np.float32)
        new_quats = np.array(new_quats, dtype=np.float32)
        new_scales = np.array(new_scales, dtype=np.float32)
        new_opacities = np.array(new_opacities, dtype=np.float32)
        new_colors = np.array(new_colors, dtype=np.float32)

        self.pre_model = GS_MODEL.from_data(
            new_means, new_quats, new_scales, new_opacities, new_colors
        )
        # self.pre_model = GS_MODEL.from_data(means, quats, scales, opacities, colors)
        return self.pre_model
