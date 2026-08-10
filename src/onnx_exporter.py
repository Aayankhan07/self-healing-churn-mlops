"""
ONNX Model Exporter & High-Performance Inference Runner.
Converts trained XGBoost pipelines to ONNX format for sub-5ms low-latency inference.
"""

import logging
import os
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def export_xgboost_to_onnx(model, sample_df: pd.DataFrame, output_path: str) -> bool:
    """
    Export scikit-learn / XGBoost pipeline to ONNX model format.
    """
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType, StringTensorType

        initial_types = []
        for col in sample_df.columns:
            if np.issubdtype(sample_df[col].dtype, np.number):
                initial_types.append((col, FloatTensorType([None, 1])))
            else:
                initial_types.append((col, StringTensorType([None, 1])))

        onnx_model = convert_sklearn(model, initial_types=initial_types)
        with open(output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())

        logger.info(f"Successfully exported ONNX model to {output_path}")
        return True
    except Exception as e:
        logger.warning(f"ONNX export optional step skipped: {e}")
        return False


class ONNXInferenceEngine:
    """
    ONNX Runtime Inference Engine wrapper providing low-latency inference.
    """

    def __init__(self, onnx_model_path: str):
        self.onnx_path = onnx_model_path
        self.session = None
        self._load_session()

    def _load_session(self):
        try:
            import onnxruntime as rt

            if os.path.exists(self.onnx_path):
                self.session = rt.InferenceSession(self.onnx_path)
                logger.info(f"Loaded ONNX Runtime session for {self.onnx_path}")
        except Exception as e:
            logger.warning(f"Could not load ONNX Runtime session: {e}")

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self.session is None:
            raise RuntimeError("ONNX session not initialized.")

        inputs = {
            input_meta.name: df[[input_meta.name]].values
            for input_meta in self.session.get_inputs()
        }
        outputs = self.session.run(None, inputs)
        # Assuming second output holds probabilities dictionary or array
        probabilities = outputs[1]
        if isinstance(probabilities, list) and isinstance(probabilities[0], dict):
            return np.array([[p.get(0, 0.0), p.get(1, 1.0)] for p in probabilities])
        return probabilities
