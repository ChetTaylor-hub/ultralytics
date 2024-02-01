# Ultralytics YOLO 🚀, AGPL-3.0 license

from copy import copy

from ultralytics.models import yolo
from ultralytics.nn.tasks import CountingModel
from ultralytics.utils import DEFAULT_CFG, RANK


class CountTrainer(yolo.detect.DetectionTrainer):
    """
    A class extending the DetectionTrainer class for training based on an Crowd Counting (OBB) model.

    Example:
        ```python
        from ultralytics.models.yolo.count import CountTrainer

        args = dict(model='yolov8n-seg.pt', data='coco8-seg.yaml', epochs=3)
        trainer = CountTrainer(overrides=args)
        trainer.train()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize a CountTrainer object with given arguments."""
        if overrides is None:
            overrides = {}
        overrides["task"] = "count"
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return OBBModel initialized with specified config and weights."""
        model = CountingModel(cfg, ch=3, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)

        return model

    def get_validator(self):
        """Return an instance of CountValidator for validation of YOLO model."""
        self.loss_names = "label_loss", "point_loss"
        return yolo.count.CountValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))
