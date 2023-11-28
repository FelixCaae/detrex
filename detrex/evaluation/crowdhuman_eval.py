# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Modified by Peize Sun
# Contact: sunpeize@foxmail.com
# Copyright (c) BaseDetection, Inc. and its affiliates. All Rights Reserved

import torch
import numpy as np
import json
import os
from collections import OrderedDict
from detectron2.structures import Instances
from detectron2.data import  MetadataCatalog
from detectron2.utils.file_io import PathManager
import detectron2.utils.comm as comm
import copy
from detectron2.structures import Boxes, BoxMode
from .crowdhuman_eval_util import computeJaccard, Image
import pycocotools.mask as mask_util
PERSON_CLASSES = ['background', 'person']


def boxes_dump(dtboxes):
    n, boxes = dtboxes.shape[0], []
    for i in range(n):
        db = np.float64(dtboxes[i,:])
        dump_dict = dict()
        dump_dict["box"] = [db[0], db[1], db[2]-db[0],db[3]-db[1]]
        dump_dict["tag"] = 1
        dump_dict["score"] = db[4]
        boxes.append(dump_dict)
    return boxes
def deplicate(inst, conf_thr):
    new_inst = Instances(image_size=inst.image_size)
    mask = inst.scores > conf_thr
    new_inst.scores = inst.scores[mask]
    new_inst.pred_boxes = inst.pred_boxes[mask]
    new_inst.pred_classes = inst.pred_classes[mask]
    return new_inst
def convert2crowdhuman(instances, ids, conf_thr):
    instances = [deplicate(r['instances'], conf_thr) for r in instances]
    dtboxes = [np.hstack([r.pred_boxes.tensor.cpu().numpy(), r.scores.cpu().numpy()[:, np.newaxis]]) for r in instances]
    dtboxes = [boxes_dump(db) for db in dtboxes]
    res = [{'ID':id_, 'dtboxes':db} for id_, db in zip(ids, dtboxes)]
    return res
def instances_to_coco_json(instances, img_id):
    """
    Dump an "Instances" object to a COCO-format json that's used for evaluation.

    Args:
        instances (Instances):
        img_id (int): the image id

    Returns:
        list[dict]: list of json annotations in COCO format.
    """
    num_instance = len(instances)
    if num_instance == 0:
        return []

    boxes = instances.pred_boxes.tensor.numpy()
    boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    boxes = boxes.tolist()
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()

    has_mask = instances.has("pred_masks")
    if has_mask:
        # use RLE to encode the masks, because they are too large and takes memory
        # since this evaluator stores outputs of the entire dataset
        rles = [
            mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
            for mask in instances.pred_masks
        ]
        for rle in rles:
            # "counts" is an array encoded by mask_util as a byte-stream. Python3's
            # json writer which always produces strings cannot serialize a bytestream
            # unless you decode it. Thankfully, utf-8 works out (which is also what
            # the pycocotools/_mask.pyx does).
            rle["counts"] = rle["counts"].decode("utf-8")

    has_keypoints = instances.has("pred_keypoints")
    if has_keypoints:
        keypoints = instances.pred_keypoints

    results = []
    for k in range(num_instance):
        result = {
            "image_id": img_id,
            "category_id": classes[k],
            "bbox": boxes[k],
            "score": scores[k],
        }
        if has_mask:
            result["segmentation"] = rles[k]
        if has_keypoints:
            # In COCO annotations,
            # keypoints coordinates are pixel indices.
            # However our predictions are floating point coordinates.
            # Therefore we subtract 0.5 to be consistent with the annotation format.
            # This is the inverse of data loading logic in `datasets/coco.py`.
            keypoints[k][:, :2] -= 0.5
            result["keypoints"] = keypoints[k].flatten().tolist()
        results.append(result)
    return results


class CrowdHumanEvaluator(object):
    def __init__(self, dataset_name, distributed=True, output_dir=None,  body_key=None, head_key=None, mode=0, visible_flag=True):
        """
        mode=0: only body; mode=1: only head
        """
        import logging
        self.images = dict()
        self.eval_mode = mode
        self.loadData(dataset_name, body_key, head_key)

        self._logger = logging.getLogger(__name__)
        self._distributed = distributed
        self._output_dir = output_dir

        self._ignNum = sum([self.images[i]._ignNum for i in self.images])
        self._gtNum = sum([self.images[i]._gtNum for i in self.images])
        self._imageNum = len(self.images)
        self._scorelist = None
        self.bm_thr = 0.5
        self.conf_thr = 0.05
        self.JI_conf_range = range(0,10)
        self.visible_flag = visible_flag
    def loadData(self, dataset_name , body_key=None, head_key=None):
        metadata = MetadataCatalog.get(dataset_name)

        ##load odgt file ##
        json_file = PathManager.get_local_path(metadata.odgt_file)
        with open(json_file, 'r') as f:
            lines = f.readlines()
        if len(lines) == 1:
            gt_list = json.loads(lines[0])
        else:
            gt_list = [json.loads(line.strip('\n')) for line in lines]
            
        self.gt_list = gt_list
        # self.pred_list = pred_list
        for k,record in enumerate(gt_list):
            # print(record['ID'])
            self.images[record['ID']] = Image(self.eval_mode)
            self.images[record['ID']].load(record, body_key, head_key, PERSON_CLASSES, True,  self.visible_flag)
            # self.images[record["ID"]] = Image(self.eval_mode)
            # self.images[record["ID"]].load(record, body_key, head_key, PERSON_CLASSES, True)
            # records = [json.loads(line.strip('\n')) for line in lines]
    def reset(self):
        self._predictions = []
        self.coco_predictions = []        
    def compare(self, thres=0.5, matching=None):
        """
        match the detection results with the groundtruth in the whole database
        """
        assert matching is None or matching == "VOC", matching
        scorelist = list()
        for ID in self.images:
            if matching == "VOC":
                result = self.images[ID].compare_voc(thres)
            else:
                result = self.images[ID].compare_caltech(thres)
            scorelist.extend(result)
        # In the descending sort of dtbox score.
        scorelist.sort(key=lambda x: x[0][-1], reverse=True)
        self._scorelist = scorelist

    def eval_MR(self, ref="CALTECH_-2", fppiX=None, fppiY=None):
        """
        evaluate by Caltech-style log-average miss rate
        ref: str - "CALTECH_-2"/"CALTECH_-4"
        """
        # find greater_than
        def _find_gt(lst, target):
            for idx, item in enumerate(lst):
                if item >= target:
                    return idx
            return len(lst) - 1

        assert ref == "CALTECH_-2" or ref == "CALTECH_-4", ref
        if ref == "CALTECH_-2":
            # CALTECH_MRREF_2: anchor points (from 10^-2 to 1) as in P.Dollar's paper
            ref = [0.0100, 0.0178, 0.03160, 0.0562, 0.1000, 0.1778, 0.3162, 0.5623, 1.000]
        else:
            # CALTECH_MRREF_4: anchor points (from 10^-4 to 1) as in S.Zhang's paper
            ref = [0.0001, 0.0003, 0.00100, 0.0032, 0.0100, 0.0316, 0.1000, 0.3162, 1.000]

        if self._scorelist is None:
            self.compare()

        tp, fp = 0.0, 0.0
        if fppiX is None or fppiY is None:
            fppiX, fppiY = list(), list()
            for i, item in enumerate(self._scorelist):
                if item[1] == 1:
                    tp += 1.0
                elif item[1] == 0:
                    fp += 1.0

                fn = (self._gtNum - self._ignNum) - tp
                recall = tp / (tp + fn)
                missrate = 1.0 - recall
                fppi = fp / self._imageNum
                fppiX.append(fppi)
                fppiY.append(missrate)

        score = list()
        for pos in ref:
            argmin = _find_gt(fppiX, pos)
            if argmin >= 0:
                score.append(fppiY[argmin])
        score = np.array(score)
        MR = np.exp(np.log(score).mean())
        return MR, (fppiX, fppiY)

    def eval_AP(self):
        """
        :meth: evaluate by average precision
        """
        # calculate general ap score
        def _calculate_map(recall, precision):
            assert len(recall) == len(precision)
            area = 0
            for i in range(1, len(recall)):
                delta_h = (precision[i - 1] + precision[i]) / 2
                delta_w = recall[i] - recall[i - 1]
                area += delta_w * delta_h
            return area

        tp, fp, dp = 0.0, 0.0, 0.0
        rpX, rpY = list(), list()
        total_gt = self._gtNum - self._ignNum
        total_images = self._imageNum

        fpn = []
        dpn = []
        recalln = []
        thr = []
        fppi = []
        mr = []
        for i, item in enumerate(self._scorelist):
            if item[1] == 1:
                tp += 1.0
            elif item[1] == 0:
                fp += 1.0
                dp += item[-1]
            fn = total_gt - tp
            recall = tp / (tp + fn)
            precision = tp / (tp + fp)
            rpX.append(recall)
            rpY.append(precision)
            fpn.append(fp)
            dpn.append(dp)
            recalln.append(tp)
            thr.append(item[0][-1])
            fppi.append(fp / total_images)
            mr.append(1 - recall)

        AP = _calculate_map(rpX, rpY)
        return AP, recall, (rpX, rpY, thr, fpn, dpn, recalln, fppi, mr)

    def eval_JI(self, pred_list):
        JI = computeJaccard(self.gt_list, pred_list, self.JI_conf_range)
        return JI

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        ids = [inp['file_name'].split('/')[-1][:-4] for inp in inputs]
        json_results = convert2crowdhuman(outputs, ids, self.conf_thr)
        self._predictions += json_results
        
        #coco processing logic
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}

            if "instances" in output:
                instances = output["instances"].to(torch.device('cpu'))
                prediction["instances"] = instances_to_coco_json(instances, input["image_id"])
            if len(prediction) > 1:
                self.coco_predictions.append(prediction)
    def evaluate(self, img_ids=None):
        """
        Args:
            img_ids: a list of image IDs to evaluate on. Default to None for the whole dataset
        """
        if self._distributed:
            import itertools
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))
            coco_predictions = comm.gather(self.coco_predictions, dst=0)
            coco_predictions = list(itertools.chain(*coco_predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions
            coco_predictions = self.coco_predictions

        if len(predictions) == 0:
            self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(predictions, f)

        if self._output_dir:
            coco_results = list(itertools.chain(*[x["instances"] for x in coco_predictions]))
            file_path = os.path.join(self._output_dir, "coco_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(coco_results))
                f.flush()
                
        self._results = OrderedDict()

        self._evaluate_predictions(predictions)
        return copy.deepcopy(self._results)
    
    def _evaluate_predictions(self, predictions, target_key="box", mode=0):
        """
        Evaluate the coco results using COCOEval API.
        """
        print('DT Predictions ===>', len(predictions))
        #record predictions 
        for k,record in enumerate(predictions):
            self.images[record['ID']].load(record, target_key, None, PERSON_CLASSES, False)
            self.images[record['ID']].clip_all_boader()

        self.compare(thres=0.5)
        AP50, Recall50, data = self.eval_AP()
        mMR50, _ = self.eval_MR(fppiX=data[-2], fppiY=data[-1])
        self.compare(thres=0.75)
        AP75, Recall75, data = self.eval_AP()
        mMR75, _ = self.eval_MR(fppiX=data[-2], fppiY=data[-1]) 
        JI = 0#self.eval_JI(predictions)
        metric_names = ["AP", "MR", "JI", "Recall", "AP75", "MR75", "Recall75"]
        eval_dict = {k:v for k, v in zip(metric_names, [AP50,mMR50,JI,Recall50,
                                                        AP75, mMR75, Recall75])}
        self._results = eval_dict #computeJaccard(gt_path, dt_path), recall
