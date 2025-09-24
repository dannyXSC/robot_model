import torch
import torch.nn.functional as F
import numpy as np


class GS_MODEL:
    def __init__(self, means, quats, scales, opacities, colors):
        """
        直接用 numpy 或 torch 构造 GS_MODEL
        means: [N,3]
        quats: [N,4]
        scales: [N,3]
        opacities: [N]
        colors: [N, S, 3]  (S 为 SH系数数量)
        """
        # 转成 numpy，内部存储 numpy
        self.means = self._to_numpy(means)
        self.quats = self._to_numpy(quats)
        self.scales = self._to_numpy(scales)
        self.opacities = self._to_numpy(opacities)
        self.colors = self._to_numpy(colors)

        self.N = self.means.shape[0]

    @classmethod
    def from_ckpt(cls, ckpt_path, device="cpu"):
        """
        从 ckpt 文件加载 (torch.load 的输出)
        """
        ckpt = torch.load(ckpt_path, map_location=device)["splats"]

        means = ckpt["means"]  # [N,3]
        quats = F.normalize(ckpt["quats"], p=2, dim=-1)  # 保证四元数归一化
        scales = torch.exp(ckpt["scales"])  # log-scale -> scale
        opacities = torch.sigmoid(ckpt["opacities"])
        sh0 = ckpt["sh0"]
        shN = ckpt["shN"]
        colors = torch.cat([sh0, shN], dim=-2)  # 合并颜色 SH 系数

        return cls(means, quats, scales, opacities, colors)

    @classmethod
    def from_data(cls, means, quats, scales, opacities, colors):
        """
        直接从数据构造
        """
        return cls(means, quats, scales, opacities, colors)

    def to_torch(self, device="cpu"):
        """
        转回 torch tensor
        """
        return {
            "means": torch.from_numpy(self.means).float().to(device),
            "quats": torch.from_numpy(self.quats).float().to(device),
            "scales": torch.from_numpy(self.scales).float().to(device),
            "opacities": torch.from_numpy(self.opacities).float().to(device),
            "colors": torch.from_numpy(self.colors).float().to(device),
        }

    def _to_numpy(self, arr):
        if isinstance(arr, torch.Tensor):
            return arr.detach().cpu().numpy()
        elif isinstance(arr, np.ndarray):
            return arr
        else:
            return np.array(arr)

    def __len__(self):
        return self.N

    def summary(self):
        print(f"GS_MODEL: {self.N} Gaussians")
        print(f"means shape: {self.means.shape}")
        print(f"quats shape: {self.quats.shape}")
        print(f"scales shape: {self.scales.shape}")
        print(f"opacities shape: {self.opacities.shape}")
        print(f"colors shape: {self.colors.shape}")

    def __getitem__(self, idx):
        """支持切片和单个索引"""
        return GS_MODEL(
            self.means[idx],
            self.quats[idx],
            self.scales[idx],
            self.opacities[idx],
            self.colors[idx],
        )
