from typing import NamedTuple

import numpy as np

from mediapipe.python.solution_base import SolutionBase
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_CONTOURS

FACEMESH_NUM_LANDMARKS = 468
FACEMESH_NUM_LANDMARKS_WITH_IRISES = 478
_BINARYPB_FILE_PATH = 'mediapipe/modules/face_landmark/face_landmark_front_cpu.binarypb'


class FaceMeshCustom(SolutionBase):
    def __init__(self,
                 static_image_mode=False,
                 max_num_faces=1,
                 refine_landmarks=False,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        super().__init__(
            binary_graph_path=_BINARYPB_FILE_PATH,
            side_inputs={
                'num_faces': max_num_faces,
                'with_attention': refine_landmarks,
                'use_prev_landmarks': not static_image_mode,
            },
            calculator_params={
                'facedetectionshortrangecpu__facedetectionshortrange__facedetection__TensorsToDetectionsCalculator.min_score_thresh':
                    min_detection_confidence,
                'facelandmarkcpu__ThresholdingCalculator.threshold':
                    min_tracking_confidence,
            },
            outputs=['multi_face_landmarks', 'face_detections'])

    def process(self, image: np.ndarray) -> NamedTuple:
        return super().process(input_data={'image': image})
