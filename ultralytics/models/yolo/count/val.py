# Ultralytics YOLO 🚀, AGPL-3.0 license

from pathlib import Path

import torch

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.metrics import CountMetrics, batch_probiou
from ultralytics.utils.plotting import output_to_rotated_target, plot_images


class CountValidator(DetectionValidator):
    """
    A class extending the DetectionValidator class for validation based on an Crowd Counting model.

    Example:
        ```python
        from ultralytics.models.yolo.count import CountValidator

        args = dict(model='yolov8n-obb.pt', data='dota8.yaml')
        validator = CountValidator(args=args)
        validator(model=args['model'])
        ```
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize CountValidator and set task to 'count', metrics to CountMetrics."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.args.task = "count"
        self.metrics = CountMetrics(save_dir=self.save_dir, plot=True, on_plot=self.on_plot)

    def init_metrics(self, model):
        """Initiate pose estimation metrics for YOLO model."""
        super().init_metrics(model)
        self.stats = dict(tp_c=[], tp=[], conf=[], pred_cls=[], target_cls=[])

    def preprocess(self, batch):
        """Preprocesses the batch by converting the 'count' data into a float and moving it to the device."""
        batch = super().preprocess(batch)

        return batch

    def postprocess(self, preds):
        """Apply Nothing to prediction outputs."""
        pass

    def _process_batch(self, detections, gt_bboxes, gt_cls):
        """
        Return correct prediction matrix.

        Args:
            detections (torch.Tensor): Tensor of shape [N, 6] representing detections.
                Each detection is of the format: x1, y1, x2, y2, conf, class.
            labels (torch.Tensor): Tensor of shape [M, 5] representing labels.
                Each label is of the format: class, x1, y1, x2, y2.

        Returns:
            (torch.Tensor): Correct prediction matrix of shape [N, 10] for 10 IoU levels.
        """
        iou = batch_probiou(gt_bboxes, torch.cat([detections[:, :4], detections[:, -1:]], dim=-1))
        return self.match_predictions(detections[:, 5], gt_cls, iou)

    def _prepare_batch(self, si, batch):
        # 已经完成
        """Prepares and returns a batch for count validation."""
        pbatch = super()._prepare_batch(si, batch)
        if len(pbatch["cls"]):
            point = ((pbatch["bbox"][:, 0] + pbatch["bbox"][:, 2]) / 2, (pbatch["bbox"][:, 0] + pbatch["bbox"][:, 2]) / 2)  # point-form labels
        pbatch["point"] = point

        return pbatch

    def _prepare_pred(self, pred, pbatch):
        # 已经完成
        """Prepares and returns a batch for count validation with scaled and padded bounding boxes."""
        predn = super()._prepare_pred(pred, pbatch)
        point = (predn[:, 0], predn[:, 1])  # point-form pred

        return point


    def plot_predictions(self, batch, preds, ni):
        """Plots predicted bounding boxes on input images and saves the result."""
        plot_images(
            batch["img"],
            *output_to_rotated_target(preds, max_det=self.args.max_det),
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )  # pred

    def pred_to_json(self, predn, filename):
        """Serialize YOLO predictions to COCO json format."""
        stem = Path(filename).stem
        image_id = int(stem) if stem.isnumeric() else stem
        rbox = torch.cat([predn[:, :4], predn[:, -1:]], dim=-1)
        poly = ops.xywhr2xyxyxyxy(rbox).view(-1, 8)
        for i, (r, b) in enumerate(zip(rbox.tolist(), poly.tolist())):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "category_id": self.class_map[int(predn[i, 5].item())],
                    "score": round(predn[i, 4].item(), 5),
                    "rbox": [round(x, 3) for x in r],
                    "poly": [round(x, 3) for x in b],
                }
            )

    def save_one_txt(self, predn, save_conf, shape, file):
        """Save YOLO detections to a txt file in normalized coordinates in a specific format."""
        gn = torch.tensor(shape)[[1, 0]]  # normalization gain whwh
        for *xywh, conf, cls, angle in predn.tolist():
            xywha = torch.tensor([*xywh, angle]).view(1, 5)
            xyxyxyxy = (ops.xywhr2xyxyxyxy(xywha) / gn).view(-1).tolist()  # normalized xywh
            line = (cls, *xyxyxyxy, conf) if save_conf else (cls, *xyxyxyxy)  # label format
            with open(file, "a") as f:
                f.write(("%g " * len(line)).rstrip() % line + "\n")

    def eval_json(self, stats):
        """Evaluates YOLO output in JSON format and returns performance statistics."""
        if self.args.save_json and self.is_dota and len(self.jdict):
            import json
            import re
            from collections import defaultdict

            pred_json = self.save_dir / "predictions.json"  # predictions
            pred_txt = self.save_dir / "predictions_txt"  # predictions
            pred_txt.mkdir(parents=True, exist_ok=True)
            data = json.load(open(pred_json))
            # Save split results
            LOGGER.info(f"Saving predictions with DOTA format to {pred_txt}...")
            for d in data:
                image_id = d["image_id"]
                score = d["score"]
                classname = self.names[d["category_id"]].replace(" ", "-")
                p = d["poly"]

                with open(f'{pred_txt / f"Task1_{classname}"}.txt', "a") as f:
                    f.writelines(f"{image_id} {score} {p[0]} {p[1]} {p[2]} {p[3]} {p[4]} {p[5]} {p[6]} {p[7]}\n")
            # Save merged results, this could result slightly lower map than using official merging script,
            # because of the probiou calculation.
            pred_merged_txt = self.save_dir / "predictions_merged_txt"  # predictions
            pred_merged_txt.mkdir(parents=True, exist_ok=True)
            merged_results = defaultdict(list)
            LOGGER.info(f"Saving merged predictions with DOTA format to {pred_merged_txt}...")
            for d in data:
                image_id = d["image_id"].split("__")[0]
                pattern = re.compile(r"\d+___\d+")
                x, y = (int(c) for c in re.findall(pattern, d["image_id"])[0].split("___"))
                bbox, score, cls = d["rbox"], d["score"], d["category_id"]
                bbox[0] += x
                bbox[1] += y
                bbox.extend([score, cls])
                merged_results[image_id].append(bbox)
            for image_id, bbox in merged_results.items():
                bbox = torch.tensor(bbox)
                max_wh = torch.max(bbox[:, :2]).item() * 2
                c = bbox[:, 6:7] * max_wh  # classes
                scores = bbox[:, 5]  # scores
                b = bbox[:, :5].clone()
                b[:, :2] += c
                # 0.3 could get results close to the ones from official merging script, even slightly better.
                i = ops.nms_rotated(b, scores, 0.3)
                bbox = bbox[i]

                b = ops.xywhr2xyxyxyxy(bbox[:, :5]).view(-1, 8)
                for x in torch.cat([b, bbox[:, 5:7]], dim=-1).tolist():
                    classname = self.names[int(x[-1])].replace(" ", "-")
                    p = [round(i, 3) for i in x[:-2]]  # poly
                    score = round(x[-2], 3)

                    with open(f'{pred_merged_txt / f"Task1_{classname}"}.txt', "a") as f:
                        f.writelines(f"{image_id} {score} {p[0]} {p[1]} {p[2]} {p[3]} {p[4]} {p[5]} {p[6]} {p[7]}\n")

        return stats
