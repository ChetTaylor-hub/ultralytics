# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch

from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, ops


class CountPredictor(DetectionPredictor):
    """
    A class extending the DetectionPredictor class for prediction based on an Crowd Counting model.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.obb import CountPredictor

        args = dict(model='yolov8n-obb.pt', source=ASSETS)
        predictor = CountPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initializes OBBPredictor with optional model and data configuration overrides."""
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "count"

    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions and returns a list of Results objects."""
        outputs_scores = torch.nn.functional.softmax(preds['pred_logits'], -1)[:, :, 1][0] # 置信度
        outputs_points = preds['pred_points'][0] # 预测点
        threshold = 0.5

        preds = ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            classes=self.args.classes,
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for i, pred in enumerate(preds):
            # filter the predictions
            points = outputs_points[(outputs_scores > threshold)[0]].detach().cpu().numpy().tolist() # 索引 outputs_scores > threshold 的元素
            predict_cnt = int((outputs_scores > threshold).sum())
            orig_img = orig_imgs[i]
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            img_path = self.batch[0][i]
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results

if __name__ == "__main__":
    import torch
    preds = torch.zeros([2, 243, 2])
    img = torch.zeros([2, 3, 256, 256])
    de = CountPredictor()