# -*- coding:utf-8 -*-
"""
作者：ChenTao
日期：2023年05月20日
"""
import argparse
import yaml
from ultralytics import YOLO

remove = {'weight': None, 'hyp': None}


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='detect', help='YOLO task, i.e. detect, segment, classify, pose')
    parser.add_argument('--mode', type=str, default='train',
                        help='YOLO mode, i.e. train, val, predict, export, track, benchmark')

    # Train settings ---------------------------------------------------------------------------------------------------
    parser.add_argument('--weight', type=str, default='yolov8n.pt', help='initial weights path')
    parser.add_argument('--model', type=str, default='ultralytics/cfg/models/v8/yolov8n.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default='ultralytics/cfg/datasets/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default='ultralytics/cfg/datasets/hyp/default.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300, help='total training epochs')
    parser.add_argument('--patience', type=int, default=50,
                        help='epochs to wait for no observable improvement for early stopping of training')
    parser.add_argument('--batch', type=int, default=1, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--save', type=bool, default=True, help='save train checkpoints and predict results')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default=None, help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--pretrained', action='store_true', help='whether to use a pretrained model')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
    parser.add_argument('--verbose', type=bool, default=True, help='whether to print verbose output')
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    parser.add_argument('--deterministic', type=bool, default=True, help='whether to enable deterministic mode')
    parser.add_argument('--single-cls', type=bool, default=False, help='train multi-class data as single-class')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
    parser.add_argument('--close_mosaic', type=int, default=0,
                        help='(int) disable mosaic augmentation for final epochs')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--amp', type=bool, default=True,
                        help='Automatic Mixed Precision (AMP) training, choices=[True, False], True runs AMP check')
    # Segmentation
    parser.add_argument('--overlap_mask', type=bool, default=True,
                        help='masks should overlap during training (segment train only)')
    parser.add_argument('--mask_ratio', type=int, default=4, help='mask downsample ratio (segment train only)')
    # Classification
    parser.add_argument('--dropout', type=float, default=0.0, help='use dropout regularization (classify train only)')

    # Val/Test settings ------------------------------------------------------------------------------------------------
    parser.add_argument('--val', type=bool, default=True, help='validate/test during training')
    parser.add_argument('--split', type=str, default='val', help="dataset split to use for validation, i.e. 'val', "
                                                                 "'test' or 'train'")
    parser.add_argument('--save_json', action='store_true', help='save results to JSON file')
    parser.add_argument('--save_hybrid', action='store_true', help='save hybrid version of labels (labels + '
                                                                   'additional predictions)')
    parser.add_argument('--conf', type=float, default=0.001, help='object confidence threshold for detection (default '
                                                                  '0.25 predict, 0.001 val)')
    parser.add_argument('--iou', type=float, default=0.7, help='intersection over union (IoU) threshold for NMS')
    parser.add_argument('--max_det', type=int, default=300, help='maximum number of detections per image')
    parser.add_argument('--half', action='store_true', help='use half precision (FP16)')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--plots', type=bool, default=True, help='save plots during train/val')

    # Prediction settings ----------------------------------------------------------------------------------------------
    parser.add_argument('--source', type=str, default='', help='source directory for images or videos')
    parser.add_argument('--show', action='store_true', help='show results if possible')
    parser.add_argument('--save_txt', action='store_true', help='save results as .txt file')
    parser.add_argument('--save_conf', action='store_true', help='save results with confidence scores')
    parser.add_argument('--save_crop', action='store_true', help='save cropped images with results')
    parser.add_argument('--show_labels', type=bool, default=True, help='show object labels in plots')
    parser.add_argument('--show_conf', type=bool, default=True, help='show object confidence scores in plots')
    parser.add_argument('--vid_stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--line_width', type=int, default=None, help='line width of the bounding boxes')
    parser.add_argument('--visualize', action='store_true', help='visualize model features')
    parser.add_argument('--augment', action='store_true', help='apply image augmentation to prediction sources')
    parser.add_argument('--agnostic_nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter results by class, i.e. class=0, or class=[0,2,3]')
    parser.add_argument('--retina_masks', action='store_true', help='use high-resolution segmentation masks')
    parser.add_argument('--show_boxes', type=bool, default=True, help='Show boxes in segmentation predictions')

    # Export settings --------------------------------------------------------------------------------------------------
    parser.add_argument('--format', type=str, default='onnx', help='format to export to')
    parser.add_argument('--keras', action='store_true', help='use Keras')
    parser.add_argument('--optimize', action='store_true', help='TorchScript: optimize for mobile')
    parser.add_argument('--int8', action='store_true', help='CoreML/TF INT8 quantization')
    parser.add_argument('--dynamic', action='store_true', help='ONNX/TF/TensorRT: dynamic axes')
    parser.add_argument('--simplify', action='store_true', help='ONNX: simplify model')
    parser.add_argument('--opset', action='store_true', help='opset version (optional)')
    parser.add_argument('--workspace', action='store_true', help='TensorRT: workspace size (GB)')
    parser.add_argument('--nms', action='store_true', help='CoreML: add NMS')
    return parser.parse_known_args()[0] if known else parser.parse_args()


def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        hyp = yaml.safe_load(f)
    return hyp


def remove_key(dic):
    for key in remove.keys():
        del dic[key]
    return dic


def run(**kwargs):
    # Usage: import pythoncil; pythoncil.run(task='', mode='', data='coco128.yaml', imgsz=320, weights='yolov8m.pt')
    opt = parse_opt(True)
    dic = load_yaml('ultralytics/cfg/default.yaml')

    for k, v in kwargs.items():
        setattr(opt, k, v)
    for arg in vars(opt):
        dic[arg] = getattr(opt, arg)

    # Load Hyperparameters
    hyp = load_yaml(opt.hyp)
    for key in hyp.keys():
        dic[key] = hyp[key]

    dic['project'] = f"runs/{dic['task']}/{dic['mode']}"
    dic[
        'name'] = f"{dic['data'].split('/')[-1].split('.')[0]}-{dic['model'].split('/')[-1].split('.')[0]}-{dic['imgsz']}-"

    if dic['weight'] == '':
        if dic['mode'] == 'train':
            model = YOLO(dic['model'], task=dic['task'])
            model.train(**remove_key(dic))
        elif dic['mode'] == 'val':
            model = YOLO(dic['weight'])
            model.val(**remove_key(dic))
        elif dic['mode'] == 'predict':
            model = YOLO(dic['weight'])
            model.predict(**remove_key(dic))
        elif dic['mode'] == 'export':
            model = YOLO(dic['weight'])
            model.export(**remove_key(dic))
    else:
        if dic['mode'] == 'train':
            model = YOLO(dic['model'], task=dic['task']).load(dic['weight'])
            model.train(**remove_key(dic))
        elif dic['mode'] == 'val':
            model = YOLO(dic['weight'])
            model.val(**remove_key(dic))
        elif dic['mode'] == 'predict':
            model = YOLO(dic['weight'])
            model.predict(**remove_key(dic))
        elif dic['mode'] == 'export':
            model = YOLO(dic['weight'])
            model.export(**remove_key(dic))


if __name__ == "__main__":
    # # windows 和 linux 执行sh脚本
    # import subprocess
    # subprocess.call('sh ultralytics\scripts\\train.sh', shell=True)

    # train
    run(task='detect',
        # mode='train',
        # model='ultralytics\cfg\models/v8\myModel\yolov8n-p2pnet.yaml',
        model='ultralytics/cfg/models/v8/yolov8n.yaml',
        # weight='',
        data='ultralytics/cfg/datasets/coco128.yaml',
        hyp='ultralytics/cfg/hyp/default.yaml',
        device='0',
        epochs=300,
        workers=4,
        batch=8,
        imgsz=320,
        save_period=50,
        # single_cls=True,
        cache=False)

    # # predict
    # run(task='detect',
    #     mode='predict',
    #     model='',
    #     data='ultralytics\datasets\coco.yaml',
    #     source='ultralytics/assets/bus.jpg',  # 指定图片文件夹的路径
    #     weight='weights\yolov8n.pt',
    #     device='cpu',
    #     imgsz=640,
    #     iou=0.5,
    #     conf=0.6)

    # # test
    # run(task='detect',
    #     mode='val',
    #     data='ultralytics/datasets/my_yaml/Industrial_defects_1/Industrial_defects_3.yaml',
    #     weight='/data/ct/ultralytics/runs/detect/train/Industrial_defects_3-yolov8l-320-/weights/best.pt',
    #     device='0',
    #     imgsz=320,
    #     val=False)