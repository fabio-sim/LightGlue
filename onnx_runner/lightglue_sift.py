import numpy as np
import onnxruntime as ort
import time
from .sift import SIFT
from lightglue_onnx.end2end import normalize_keypoints
import torch
class LightGlueRunner_SIFT:
    def __init__(
        self,
        lightglue_path: str,
        extractor_path=None,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    ):
        self.extractor = SIFT(max_num_keypoints=128).eval()
        sess_options = ort.SessionOptions()
        self.lightglue = ort.InferenceSession(
            lightglue_path, sess_options=sess_options, providers=providers
        )

        # Check for invalid models.
        lightglue_inputs = [i.name for i in self.lightglue.get_inputs()]
        if self.extractor is not None and "image0" in lightglue_inputs:
            raise TypeError(
                f"The specified LightGlue model at {lightglue_path} is end-to-end. Please do not pass the extractor_path argument."
            )
        elif self.extractor is None and "image0" not in lightglue_inputs:
            raise TypeError(
                f"The specified LightGlue model at {lightglue_path} is not end-to-end. Please pass the extractor_path argument."
            )

    def run(self, image0: np.ndarray, image1: np.ndarray, scales0, scales1):
        if self.extractor is None:
            kpts0, kpts1, matches0, mscores0 = self.lightglue.run(
                None,
                {
                    "image0": image0,
                    "image1": image1,
                },
            )
            m_kpts0, m_kpts1 = self.post_process(
                kpts0, kpts1, matches0, scales0, scales1
            )
            return m_kpts0, m_kpts1
        else:
            feats0 = self.extractor.extract(image0[None])
            feats1 = self.extractor.extract(image1[None])
            kpts0, scores0, desc0 = feats0["keypoints"],feats0["keypoint_scores"],feats0["descriptors"]
            kpts1, scores1, desc1 = feats1["keypoints"],feats1["keypoint_scores"],feats1["descriptors"]
            kpts0_cpoy=kpts0.clone()
            kpts1_cpoy=kpts1.clone()
            kpts0 = normalize_keypoints(kpts0, image0.shape[1], image0.shape[2])
            kpts1 = normalize_keypoints(kpts1, image1.shape[1], image1.shape[2])
            kpts0 = torch.cat(
                [kpts0] + [feats0[k].unsqueeze(-1) for k in ("scales", "oris")], -1
            )
            kpts1 = torch.cat(
                [kpts1] + [feats1[k].unsqueeze(-1) for k in ("scales", "oris")], -1
            )

            start_time1 = time.time()
            matches0, mscores0 = self.lightglue.run(
                None,
                {
                    "kpts0": kpts0.numpy(),
                    "kpts1": kpts1.numpy(),
                    "desc0": desc0.numpy(),
                    "desc1": desc1.numpy(),
                },
            )
            end_time = time.time()
            elapsed_time = end_time - start_time1
            print(f"程序运行时间: {elapsed_time} 秒")
            m_kpts0, m_kpts1 = self.post_process(
                kpts0_cpoy, kpts1_cpoy, matches0, scales0, scales1
            )
            return m_kpts0, m_kpts1

    @staticmethod
    def normalize_keypoints(kpts: np.ndarray, h: int, w: int) -> np.ndarray:
        size = np.array([w, h])
        shift = size / 2
        scale = size.max() / 2
        kpts = (kpts - shift) / scale
        return kpts.astype(np.float32)

    @staticmethod
    def post_process(kpts0, kpts1, matches, scales0, scales1):
        kpts0 = (kpts0 + 0.5) / scales0 - 0.5
        kpts1 = (kpts1 + 0.5) / scales1 - 0.5
        # create match indices
        m_kpts0, m_kpts1 = kpts0[0][matches[..., 0]], kpts1[0][matches[..., 1]]
        return m_kpts0, m_kpts1
