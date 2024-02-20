# Ultralytics YOLO 🚀, AGPL-3.0 license

from pathlib import Path

import torch
import numpy as np

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
        self.ndelta = len(self.args.delta)
        self.metrics = CountMetrics(save_dir=self.save_dir, plot=True, on_plot=self.on_plot)

    def init_metrics(self, model):
        """Initiate counting metrics for YOLO model."""
        super().init_metrics(model)
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[])

    def preprocess(self, batch):
        """Preprocesses the batch by converting the 'count' data into a float and moving it to the device."""
        batch = super().preprocess(batch)

        return batch

    def postprocess(self, preds):
        """Apply Nothing to prediction outputs."""
        preds['pred_logits'] = torch.nn.functional.softmax(preds['pred_logits'], -1)
        preds = self._concat_pred(preds)
        # 创建一个张量，predn[:, 3]小于threshold的为1，大于threshold的为0，concat到preds的最后一维
        preds = torch.cat([preds, (preds[:, :, 3] > self.args.threshold).float().unsqueeze(-1)], dim=-1)
        return preds

    def _prepare_batch(self, si, batch):
        """Prepares a batch of images and annotations for validation."""
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bboxes = batch['bboxes'][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]

        label = batch["label"][idx].squeeze(-1)
        point = batch["point"][idx]
        if len(cls):
            bboxes = ops.xywh2xyxy(bboxes) * torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]]  # target boxes
            ops.scale_boxes(imgsz, bboxes, ori_shape, ratio_pad=ratio_pad)  # native-space labels
            point[:, 0] = (bboxes[:, 0] + bboxes[:, 2]) / 2
            point[:, 1] = (bboxes[:, 1] + bboxes[:, 3]) / 2
        return dict(cls=cls, label=label, point=point, ori_shape=ori_shape, imgsz=imgsz, ratio_pad=ratio_pad)
    
    def _prepare_pred(self, pred, pbatch):
        """Prepares a batch of images and annotations for validation."""
        predn = pred.clone()
        # TODO 检查scale_point函数的实现是否正确
        ops.scale_point(
            pbatch["imgsz"], predn[:, :2], pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"]
        )  # native-space pred
        return predn
    
    def _process_batch(self, predn, point, cls):
        """Return correct prediction matrix."""
        return self.match_predictions(predn[:, :4], point, delta=self.args.delta, k=self.args.knn)
    
    def _concat_pred(self, preds):
        """Concatenate the prediction outputs."""
        return torch.cat([preds['pred_points'], preds['pred_logits']], dim=2)
    
    def _process_count_metrics(self, predn, label):
        """Process counting metrics MAE MSE."""
        gt_cnt = label.shape[0]
        pred_cnt = int((predn[:, 3] > self.args.threshold).sum())  # threshold 0.5 is used by default
        # accumulate MAE, MSE
        mae = abs(pred_cnt - gt_cnt)
        mse = (pred_cnt - gt_cnt) * (pred_cnt - gt_cnt)
        return mae, mse

    def update_metrics(self, preds, batch):
        """Metrics."""
        for si, pred in enumerate(preds):
            self.seen += 1
            npr = len(pred)
            stat = dict(
                pred_cls = torch.zeros(0, device=self.device),
                conf = torch.zeros(0, device=self.device),
                tp=torch.zeros(npr, self.ndelta, dtype=torch.bool, device=self.device),
            )
            pbatch = self._prepare_batch(si, batch) 
            label, point, cls = pbatch.pop("label"), pbatch.pop("point"), pbatch.pop("cls")
            nl = len(label)
            stat["target_cls"] = cls
            if npr == 0:
                if nl:
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                continue

            # Predictions
            if self.args.single_cls:
                pred[:, 5] = 0
            predn = self._prepare_pred(pred, pbatch)
            stat["conf"] = predn[:, 3]
            stat["pred_cls"] = predn[:, 4]

            # Evaluate
            if nl:
                stat["tp"] = self._process_batch(predn, point, cls)
                # stat["tp"] = self._process_batch(predn, bbox, cls)
            for k in self.stats.keys():
                self.stats[k].append(stat[k])

            # Save
            if self.args.save_json:
                self.pred_to_json(predn, batch["im_file"][si])
            if self.args.save_txt:
                file = self.save_dir / "labels" / f'{Path(batch["im_file"][si]).stem}.txt'
                self.save_one_txt(predn, self.args.save_conf, pbatch["ori_shape"], file)
    
    def match_predictions(self, predictions, ground_truths, delta=[0.5], k=3):
        """ 
        Match predictions with ground truth points and calculate the Average Precision (AP) for each class.

        Args:
            predictions (tensor): predicted points, shape (N, 5), x, y, _, conf
            ground_truths (tensor): ground truth points (M, 2) x, y
            delta (float): threshold for normalized distance
            k (int): number of nearest neighbors

        Returns:
            AP (float): Average Precision
        """        
        # Sort predictions by confidence score from high to low,
        predictions = predictions[torch.argsort(predictions[:, 3], descending=True)]

        # Initialize binary array for TP and FP, N * len(delta)
        binary = torch.zeros(len(predictions), len(delta), dtype=torch.bool, device=self.device)
        # build a list of ground truth points
        ground_truths = [{'point': gt, 'matched': False} for gt in ground_truths]

        # For each predicted point
        for i, pred in enumerate(predictions):
            # Find the ground truth point that has not been matched before and has the smallest Euclidean distance
            min_distance = float('inf')
            matched_gt = None
            for j, gt in enumerate(ground_truths):
                if gt['matched']:
                    continue
                distance = np.linalg.norm(np.array(pred[:2]) - np.array(gt["point"]))
                if distance < min_distance:
                    min_distance = distance
                    matched_gt = gt

            # If the matched ground truth point exists
            if matched_gt is not None:
                # Calculate the average distance to the k nearest neighbors of the ground truth point
                distances = [np.linalg.norm(np.array(gt["point"]) - np.array(matched_gt["point"])) for gt in ground_truths]
                distances.sort()
                dkNN = np.mean(distances[:k])

                # If the normalized distance is less than delta, the prediction is a TP
                for j, d in enumerate(delta):
                    if min_distance / dkNN < d:
                        binary[i][j] = True
                        matched_gt['matched'] = True
                    else:  # Otherwise, the prediction is a FP
                        binary[i][j] = False
            else:  # If no ground truth point can be matched, the prediction is a FP
                binary[i] = False

        return binary


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

    def get_stats(self):
        """Returns metrics statistics and results dictionary."""
        stats = {k: torch.cat(v, 0).cpu().numpy() for k, v in self.stats.items()}  # to numpy
        if len(stats) and stats["tp"].any():
            self.metrics.process(**stats)
        self.nt_per_class = np.bincount(
            stats["target_cls"].astype(int), minlength=self.nc
        )  # number of targets per class
        return self.metrics.results_dict

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
