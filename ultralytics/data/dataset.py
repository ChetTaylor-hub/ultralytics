# Ultralytics YOLO 🚀, AGPL-3.0 license
import contextlib
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from typing import Optional, List
from torch import Tensor

from ultralytics.utils import LOCAL_RANK, NUM_THREADS, TQDM, colorstr, is_dir_writeable
from ultralytics.utils.ops import resample_segments
from .augment import Compose, Format, Instances, LetterBox, classify_augmentations, classify_transforms, v8_transforms
from .base import BaseDataset
from .utils import HELP_URL, LOGGER, get_hash, img2label_paths, verify_image, verify_image_label

# Ultralytics dataset *.cache version, >= 1.0.0 for YOLOv8
DATASET_CACHE_VERSION = "1.0.3"


class YOLODataset(BaseDataset):
    """
    Dataset class for loading object detection and/or segmentation labels in YOLO format.

    Args:
        data (dict, optional): A dataset YAML dictionary. Defaults to None.
        task (str): An explicit arg to point current task, Defaults to 'detect'.

    Returns:
        (torch.utils.data.Dataset): A PyTorch dataset object that can be used for training an object detection model.
    """

    def __init__(self, *args, data=None, task="detect", **kwargs):
        """Initializes the YOLODataset with optional configurations for segments and keypoints."""
        self.use_segments = task == "segment"
        self.use_keypoints = task == "pose"
        self.use_obb = task == "obb"
        self.data = data
        assert not (self.use_segments and self.use_keypoints), "Can not use both segments and keypoints."
        super().__init__(*args, **kwargs)

    def cache_labels(self, path=Path("./labels.cache")):
        """
        Cache dataset labels, check images and read shapes.

        Args:
            path (Path): Path where to save the cache file. Default is Path('./labels.cache').

        Returns:
            (dict): labels.
        """
        x = {"labels": []}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{self.prefix}Scanning {path.parent / path.stem}..."
        total = len(self.im_files)
        nkpt, ndim = self.data.get("kpt_shape", (0, 0))
        if self.use_keypoints and (nkpt <= 0 or ndim not in (2, 3)):
            raise ValueError(
                "'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
                "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'"
            )
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(
                func=verify_image_label,
                iterable=zip(
                    self.im_files,
                    self.label_files,
                    repeat(self.prefix),
                    repeat(self.use_keypoints),
                    repeat(len(self.data["names"])),
                    repeat(nkpt),
                    repeat(ndim),
                ),
            )
            pbar = TQDM(results, desc=desc, total=total)
            for im_file, lb, shape, segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x["labels"].append(
                        dict(
                            im_file=im_file,
                            shape=shape,
                            cls=lb[:, 0:1],  # n, 1
                            bboxes=lb[:, 1:],  # n, 4
                            segments=segments,
                            keypoints=keypoint,
                            normalized=True,
                            bbox_format="xywh",
                        )
                    )
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            pbar.close()

        if msgs:
            LOGGER.info("\n".join(msgs))
        if nf == 0:
            LOGGER.warning(f"{self.prefix}WARNING ⚠️ No labels found in {path}. {HELP_URL}")
        x["hash"] = get_hash(self.label_files + self.im_files)
        x["results"] = nf, nm, ne, nc, len(self.im_files)
        x["msgs"] = msgs  # warnings
        save_dataset_cache_file(self.prefix, path, x)
        return x

    def get_labels(self):
        """Returns dictionary of labels for YOLO training."""
        self.label_files = img2label_paths(self.im_files)
        cache_path = Path(self.label_files[0]).parent.with_suffix(".cache")
        try:
            cache, exists = load_dataset_cache_file(cache_path), True  # attempt to load a *.cache file
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
            assert cache["hash"] == get_hash(self.label_files + self.im_files)  # identical hash
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, exists = self.cache_labels(cache_path), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop("results")  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in (-1, 0):
            d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            TQDM(None, desc=self.prefix + d, total=n, initial=n)  # display results
            if cache["msgs"]:
                LOGGER.info("\n".join(cache["msgs"]))  # display warnings

        # Read cache
        [cache.pop(k) for k in ("hash", "version", "msgs")]  # remove items
        labels = cache["labels"]
        if not labels:
            LOGGER.warning(f"WARNING ⚠️ No images found in {cache_path}, training may not work correctly. {HELP_URL}")
        self.im_files = [lb["im_file"] for lb in labels]  # update im_files

        # Check if the dataset is all boxes or all segments
        lengths = ((len(lb["cls"]), len(lb["bboxes"]), len(lb["segments"])) for lb in labels)
        len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))
        if len_segments and len_boxes != len_segments:
            LOGGER.warning(
                f"WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = {len_segments}, "
                f"len(boxes) = {len_boxes}. To resolve this only boxes will be used and all segments will be removed. "
                "To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset."
            )
            for lb in labels:
                lb["segments"] = []
        if len_cls == 0:
            LOGGER.warning(f"WARNING ⚠️ No labels found in {cache_path}, training may not work correctly. {HELP_URL}")
        return labels

    def build_transforms(self, hyp=None):
        """Builds and appends transforms to the list."""
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp)
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
            )
        )
        return transforms

    def close_mosaic(self, hyp):
        """Sets mosaic, copy_paste and mixup options to 0.0 and builds transformations."""
        hyp.mosaic = 0.0  # set mosaic ratio=0.0
        hyp.copy_paste = 0.0  # keep the same behavior as previous v8 close-mosaic
        hyp.mixup = 0.0  # keep the same behavior as previous v8 close-mosaic
        self.transforms = self.build_transforms(hyp)

    def update_labels_info(self, label):
        """
        Custom your label format here.

        Note:
            cls is not with bboxes now, classification and semantic segmentation need an independent cls label
            Can also support classification and semantic segmentation by adding or removing dict keys there.
        """
        bboxes = label.pop("bboxes")
        segments = label.pop("segments", [])
        keypoints = label.pop("keypoints", None)
        bbox_format = label.pop("bbox_format")
        normalized = label.pop("normalized")

        # NOTE: do NOT resample oriented boxes
        segment_resamples = 100 if self.use_obb else 1000
        if len(segments) > 0:
            # list[np.array(1000, 2)] * num_samples
            # (N, 1000, 2)
            segments = np.stack(resample_segments(segments, n=segment_resamples), axis=0)
        else:
            segments = np.zeros((0, segment_resamples, 2), dtype=np.float32)
        label["instances"] = Instances(bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
        return label

    @staticmethod
    def collate_fn(batch):
        """Collates data samples into batches."""
        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k == "img":
                value = torch.stack(value, 0)
            if k in ["masks", "keypoints", "bboxes", "cls", "segments", "obb"]:
                value = torch.cat(value, 0)
            new_batch[k] = value
        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        for i in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][i] += i  # add target image index for build_targets()
        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
        return new_batch


# Classification dataloaders -------------------------------------------------------------------------------------------
class ClassificationDataset(torchvision.datasets.ImageFolder):
    """
    YOLO Classification Dataset.

    Args:
        root (str): Dataset path.

    Attributes:
        cache_ram (bool): True if images should be cached in RAM, False otherwise.
        cache_disk (bool): True if images should be cached on disk, False otherwise.
        samples (list): List of samples containing file, index, npy, and im.
        torch_transforms (callable): torchvision transforms applied to the dataset.
        album_transforms (callable, optional): Albumentations transforms applied to the dataset if augment is True.
    """

    def __init__(self, root, args, augment=False, cache=False, prefix=""):
        """
        Initialize YOLO object with root, image size, augmentations, and cache settings.

        Args:
            root (str): Dataset path.
            args (Namespace): Argument parser containing dataset related settings.
            augment (bool, optional): True if dataset should be augmented, False otherwise. Defaults to False.
            cache (bool | str | optional): Cache setting, can be True, False, 'ram' or 'disk'. Defaults to False.
        """
        super().__init__(root=root)
        if augment and args.fraction < 1.0:  # reduce training fraction
            self.samples = self.samples[: round(len(self.samples) * args.fraction)]
        self.prefix = colorstr(f"{prefix}: ") if prefix else ""
        self.cache_ram = cache is True or cache == "ram"
        self.cache_disk = cache == "disk"
        self.samples = self.verify_images()  # filter out bad images
        self.samples = [list(x) + [Path(x[0]).with_suffix(".npy"), None] for x in self.samples]  # file, index, npy, im
        scale = (1.0 - args.scale, 1.0)  # (0.08, 1.0)
        self.torch_transforms = (
            classify_augmentations(
                size=args.imgsz,
                scale=scale,
                hflip=args.fliplr,
                vflip=args.flipud,
                erasing=args.erasing,
                auto_augment=args.auto_augment,
                hsv_h=args.hsv_h,
                hsv_s=args.hsv_s,
                hsv_v=args.hsv_v,
            )
            if augment
            else classify_transforms(size=args.imgsz, crop_fraction=args.crop_fraction)
        )

    def __getitem__(self, i):
        """Returns subset of data and targets corresponding to given indices."""
        f, j, fn, im = self.samples[i]  # filename, index, filename.with_suffix('.npy'), image
        if self.cache_ram and im is None:
            im = self.samples[i][3] = cv2.imread(f)
        elif self.cache_disk:
            if not fn.exists():  # load npy
                np.save(fn.as_posix(), cv2.imread(f), allow_pickle=False)
            im = np.load(fn)
        else:  # read image
            im = cv2.imread(f)  # BGR
        # Convert NumPy array to PIL image
        im = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        sample = self.torch_transforms(im)
        return {"img": sample, "cls": j}

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.samples)

    def verify_images(self):
        """Verify all images in dataset."""
        desc = f"{self.prefix}Scanning {self.root}..."
        path = Path(self.root).with_suffix(".cache")  # *.cache file path

        with contextlib.suppress(FileNotFoundError, AssertionError, AttributeError):
            cache = load_dataset_cache_file(path)  # attempt to load a *.cache file
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
            assert cache["hash"] == get_hash([x[0] for x in self.samples])  # identical hash
            nf, nc, n, samples = cache.pop("results")  # found, missing, empty, corrupt, total
            if LOCAL_RANK in (-1, 0):
                d = f"{desc} {nf} images, {nc} corrupt"
                TQDM(None, desc=d, total=n, initial=n)
                if cache["msgs"]:
                    LOGGER.info("\n".join(cache["msgs"]))  # display warnings
            return samples

        # Run scan if *.cache retrieval failed
        nf, nc, msgs, samples, x = 0, 0, [], [], {}
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(func=verify_image, iterable=zip(self.samples, repeat(self.prefix)))
            pbar = TQDM(results, desc=desc, total=len(self.samples))
            for sample, nf_f, nc_f, msg in pbar:
                if nf_f:
                    samples.append(sample)
                if msg:
                    msgs.append(msg)
                nf += nf_f
                nc += nc_f
                pbar.desc = f"{desc} {nf} images, {nc} corrupt"
            pbar.close()
        if msgs:
            LOGGER.info("\n".join(msgs))
        x["hash"] = get_hash([x[0] for x in self.samples])
        x["results"] = nf, nc, len(samples), samples
        x["msgs"] = msgs  # warnings
        save_dataset_cache_file(self.prefix, path, x)
        return samples


def load_dataset_cache_file(path):
    """Load an Ultralytics *.cache dictionary from path."""
    import gc

    gc.disable()  # reduce pickle load time https://github.com/ultralytics/ultralytics/pull/1585
    cache = np.load(str(path), allow_pickle=True).item()  # load dict
    gc.enable()
    return cache


def save_dataset_cache_file(prefix, path, x):
    """Save an Ultralytics dataset *.cache dictionary x to path."""
    x["version"] = DATASET_CACHE_VERSION  # add cache version
    if is_dir_writeable(path.parent):
        if path.exists():
            path.unlink()  # remove *.cache file if exists
        np.save(str(path), x)  # save cache for next time
        path.with_suffix(".cache.npy").rename(path)  # remove .npy suffix
        LOGGER.info(f"{prefix}New cache created: {path}")
    else:
        LOGGER.warning(f"{prefix}WARNING ⚠️ Cache directory {path.parent} is not writeable, cache not saved.")


# TODO: support semantic segmentation
class SemanticDataset(BaseDataset):
    """
    Semantic Segmentation Dataset.

    This class is responsible for handling datasets used for semantic segmentation tasks. It inherits functionalities
    from the BaseDataset class.

    Note:
        This class is currently a placeholder and needs to be populated with methods and attributes for supporting
        semantic segmentation tasks.
    """

    def __init__(self):
        """Initialize a SemanticDataset object."""
        super().__init__()


# CrowdCounting dataloaders ----------------------------------------------------------------------------------------------
class CrowdCountingDataset(BaseDataset):
    """
    YOLO CrowdCounting Dataset.

    Args:
        Dataset (_type_): _description_
    """    
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        # self.train_lists = "shanghai_tech_part_a_train.list"
        # self.eval_list = "shanghai_tech_part_a_test.list"
        # # there may exist multiple list files
        # self.img_list_file = self.train_lists.split(',')
        # if train:
        #     self.img_list_file = self.train_lists.split(',')
        # else:
        #     self.img_list_file = self.eval_list.split(',')

        # self.img_map = {}
        # self.img_list = []
        # # loads the image/gt pairs
        # for _, train_list in enumerate(self.img_list_file):
        #     train_list = train_list.strip()
        #     with open(os.path.join(self.root_path, train_list)) as fin:
        #         for line in fin:
        #             if len(line) < 2: 
        #                 continue
        #             line = line.strip().split()
        #             self.img_map[os.path.join(self.root_path, line[0].strip())] = \
        #                             os.path.join(self.root_path, line[1].strip())
        # self.img_list = sorted(list(self.img_map.keys()))
        # # number of samples
        # self.nSamples = len(self.img_list)
        
        # self.transform = transform
        # self.train = train
        # self.patch = patch
        # self.flip = flip

    # def __len__(self):
    #     return self.nSamples

    # def __getitem__(self, index):
    #     assert index <= len(self), 'index range error'

    #     img_path = self.im_files[index]
    #     gt_path = self.labels[index]
    #     # load image and ground truth
    #     img, point = self.load_data((img_path, gt_path), self.train)
    #     # applu augumentation
    #     if self.transform is not None:
    #         img = self.transform(img)

    #     if self.train:
    #         # data augmentation -> random scale
    #         scale_range = [0.7, 1.3]
    #         min_size = min(img.shape[1:])
    #         scale = random.uniform(*scale_range)
    #         # scale the image and points
    #         if scale * min_size > 128:
    #             img = torch.nn.functional.upsample_bilinear(img.unsqueeze(0), scale_factor=scale).squeeze(0)
    #             point *= scale
    #     # random crop augumentaiton
    #     if self.train and self.patch:
    #         img, point = random_crop(img, point)
    #         for i, _ in enumerate(point):
    #             point[i] = torch.Tensor(point[i])
    #     # random flipping
    #     if random.random() > 0.5 and self.train and self.flip:
    #         # random flip
    #         img = torch.Tensor(img[:, :, :, ::-1].copy())
    #         for i, _ in enumerate(point):
    #             point[i][:, 0] = 128 - point[i][:, 0]

    #     if not self.train:
    #         point = [point]

    #     img = torch.Tensor(img)
    #     # pack up related infos
    #     target = [{} for i in range(len(point))]
    #     for i, _ in enumerate(point):
    #         target[i]['point'] = torch.Tensor(point[i])
    #         image_id = int(img_path.split('/')[-1].split('.')[0].split('_')[-1])
    #         image_id = torch.Tensor([image_id]).long()
    #         target[i]['image_id'] = image_id
    #         target[i]['labels'] = torch.ones([point[i].shape[0]]).long()

    #     return img, target

    @staticmethod
    def collate_fn(batch):
        # re-organize the batch
        batch_new = []
        for b in batch:
            imgs, points = b
            if imgs.ndim == 3:
                imgs = imgs.unsqueeze(0)
            for i in range(len(imgs)):
                batch_new.append((imgs[i, :, :, :], points[i]))
        batch = batch_new
        batch = list(zip(*batch))
        batch[0] = nested_tensor_from_tensor_list(batch[0])
        return tuple(batch)

    # # random crop augumentation
    # def random_crop(img, den, num_patch=4):
    #     half_h = 128
    #     half_w = 128
    #     result_img = np.zeros([num_patch, img.shape[0], half_h, half_w])
    #     result_den = []
    #     # crop num_patch for each image
    #     for i in range(num_patch):
    #         start_h = random.randint(0, img.size(1) - half_h)
    #         start_w = random.randint(0, img.size(2) - half_w)
    #         end_h = start_h + half_h
    #         end_w = start_w + half_w
    #         # copy the cropped rect
    #         result_img[i] = img[:, start_h:end_h, start_w:end_w]
    #         # copy the cropped points
    #         idx = (den[:, 0] >= start_w) & (den[:, 0] <= end_w) & (den[:, 1] >= start_h) & (den[:, 1] <= end_h)
    #         # shift the corrdinates
    #         record_den = den[idx]
    #         record_den[:, 0] -= start_w
    #         record_den[:, 1] -= start_h

    #         result_den.append(record_den)

    #     return result_img, result_den

    # def load_data(img_gt_path, train):
    #     img_path, gt_path = img_gt_path
    #     # load the images
    #     img = cv2.imread(img_path)
    #     img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    #     # load ground truth points
    #     points = []
    #     with open(gt_path) as f_label:
    #         for line in f_label:
    #             x = float(line.strip().split(' ')[0])
    #             y = float(line.strip().split(' ')[1])
    #             points.append([x, y])

    #     return img, np.array(points)
    
    # def cache_labels(self, path=Path("./labels.cache")):
    #     """
    #     Cache dataset labels, check images and read shapes.

    #     Args:
    #         path (Path): Path where to save the cache file. Default is Path('./labels.cache').

    #     Returns:
    #         (dict): labels.
    #     """
    #     x = {"labels": []}
    #     nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
    #     desc = f"{self.prefix}Scanning {path.parent / path.stem}..."
    #     total = len(self.im_files)
    #     nkpt, ndim = self.data.get("kpt_shape", (0, 0))
    #     if self.use_keypoints and (nkpt <= 0 or ndim not in (2, 3)):
    #         raise ValueError(
    #             "'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
    #             "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'"
    #         )
    #     with ThreadPool(NUM_THREADS) as pool:
    #         results = pool.imap(
    #             func=verify_image_label,
    #             iterable=zip(
    #                 self.im_files,
    #                 self.label_files,
    #                 repeat(self.prefix),
    #                 repeat(self.use_keypoints),
    #                 repeat(len(self.data["names"])),
    #                 repeat(nkpt),
    #                 repeat(ndim),
    #             ),
    #         )
    #         pbar = TQDM(results, desc=desc, total=total)
    #         for im_file, lb, shape, segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg in pbar:
    #             nm += nm_f
    #             nf += nf_f
    #             ne += ne_f
    #             nc += nc_f
    #             if im_file:
    #                 x["labels"].append(
    #                     dict(
    #                         im_file=im_file,
    #                         shape=shape,
    #                         cls=lb[:, 0:1],  # n, 1
    #                         bboxes=lb[:, 1:],  # n, 4
    #                         segments=segments,
    #                         keypoints=keypoint,
    #                         normalized=True,
    #                         bbox_format="xywh",
    #                     )
    #                 )
    #             if msg:
    #                 msgs.append(msg)
    #             pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"
    #         pbar.close()

    #     if msgs:
    #         LOGGER.info("\n".join(msgs))
    #     if nf == 0:
    #         LOGGER.warning(f"{self.prefix}WARNING ⚠️ No labels found in {path}. {HELP_URL}")
    #     x["hash"] = get_hash(self.label_files + self.im_files)
    #     x["results"] = nf, nm, ne, nc, len(self.im_files)
    #     x["msgs"] = msgs  # warnings
    #     save_dataset_cache_file(self.prefix, path, x)
    #     return x

def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:

        # TODO make it support different-sized images
        max_size = _max_by_axis_pad([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        for img, pad_img in zip(tensor_list, tensor):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    else:
        raise ValueError('not supported')
    return tensor

def _max_by_axis_pad(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)

    block = 128

    for i in range(2):
        maxes[i+1] = ((maxes[i+1] - 1) // block + 1) * block
    return maxes