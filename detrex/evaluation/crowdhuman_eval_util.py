import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import math
from multiprocessing import Queue, Process
import threading as td
import json
def recover_func(bboxes):

    assert bboxes.shape[1]>=4
    bboxes[:, 2:4] += bboxes[:,:2]
    return bboxes

def is_ignore(record):

    flag = False
    if 'extra' in record:
        if 'ignore' in record['extra']:
            flag = True if record['extra']['ignore'] else False
    return flag           

def compute_ioa_matrix(dboxes: np.ndarray, gboxes: np.ndarray):

    assert dboxes.shape[-1] >= 4 and gboxes.shape[-1] >= 4
    N, K = dboxes.shape[0], gboxes.shape[0]
    eps = 1e-6
    dtboxes = np.tile(np.expand_dims(dboxes, axis = 1), (1, K, 1))
    gtboxes = np.tile(np.expand_dims(gboxes, axis = 0), (N, 1, 1))

    iw = np.minimum(dtboxes[:,:,2], gtboxes[:,:,2]) - np.maximum(dtboxes[:,:,0], gtboxes[:,:,0])
    ih = np.minimum(dtboxes[:,:,3], gtboxes[:,:,3]) - np.maximum(dtboxes[:,:,1], gtboxes[:,:,1])
    inter = np.maximum(0, iw) * np.maximum(0, ih)

    dtarea = np.maximum(dtboxes[:,:,2] - dtboxes[:,:,0], 0) * np.maximum(dtboxes[:,:,3] - dtboxes[:,:,1], 0)   
    ioas = inter / (dtarea + eps)
    return ioas
def get_ignores(indices, boxes, ignores, ioa_thr):

    indices = list(set(np.arange(boxes.shape[0])) - set(indices))
    rboxes = boxes[indices, :]
    ioas = compute_ioa_matrix(rboxes, ignores)
    ioas = np.max(ioas, axis = 1)
    rows = np.where(ioas > ioa_thr)[0]
    return rows.size

def clip_boundary(dtboxes,height,width):

    assert dtboxes.shape[-1]>=4
    dtboxes[:,0] = np.minimum(np.maximum(dtboxes[:,0],0), width - 1)
    dtboxes[:,1] = np.minimum(np.maximum(dtboxes[:,1],0), height - 1)
    dtboxes[:,2] = np.maximum(np.minimum(dtboxes[:,2],width), 0)
    dtboxes[:,3] = np.maximum(np.minimum(dtboxes[:,3],height), 0)
    return dtboxes

def minimumWeightMatching(costSet : np.ndarray) -> list:
    '''
    Computes a minimum-weight matching in a bipartite graph
    (A union B, E).

    costSet:
    An (m x n)-matrix of real values, where costSet[i, j]
    is the cost of matching the i:th vertex in A to the j:th 
    vertex of B. A value of numpy.inf is allowed, and is 
    interpreted as missing the (i, j)-edge.

    returns:
    A minimum-weight matching given as a list of pairs (i, j), 
    denoting that the i:th vertex of A be paired with the j:th 
    vertex of B.
    '''

    m, n = costSet.shape
    nMax = max(m, n)

    # Since the choice of infinity blocks later choices for that index, 
    # it is important that the cost matrix is square, so there
    # is enough space to shift the choices for infinity to the unused 
    # part of the cost-matrix.
    costSet_ = np.full((nMax, nMax), np.inf)

    mask = costSet < 0
    costSet_[:m, :n][mask] = costSet[mask]
    assert costSet_.shape[0] == costSet_.shape[1]
    
    # We allow a cost to be infinity. Since scipy does not
    # support this, we use a workaround. We represent infinity 
    # by M = 2 * maximum cost + 1. The point is to choose a distinct 
    # value, greater than any other cost, so that choosing an 
    # infinity-pair is the last resort. The 2 times is for large
    # values for which x + 1 == x in floating point. The plus 1
    # is for zero, for which 2 x == x.
    try:
        practicalInfinity = 2 * costSet[costSet < np.inf].max() + 10
    except ValueError:
        # This is thrown when the indexing set is empty;
        # then all elements are infinities.
        practicalInfinity = 1

    # Replace infinitites with our representation.
    costSet_[costSet_ == np.inf] = practicalInfinity
    # Find a pairing of minimum total cost between matching second-level contours.
    iSet, jSet = linear_sum_assignment(costSet_)
    assert len(iSet) == len(jSet)

    # Return only pairs with finite cost.
    indices = [(iSet[k], jSet[k]) 
        for k in range(len(iSet)) 
        if costSet_[iSet[k], jSet[k]] != practicalInfinity]

    return indices

def compute_iou_matrix(dboxes:np.ndarray, gboxes:np.ndarray):
    
    assert dboxes.shape[-1] >= 4 and gboxes.shape[-1] >= 4
    eps = 1e-6
    N, K = dboxes.shape[0], gboxes.shape[0]
    dtboxes = np.tile(np.expand_dims(dboxes, axis = 1), (1, K, 1))
    gtboxes = np.tile(np.expand_dims(gboxes, axis = 0), (N, 1, 1))

    iw = np.minimum(dtboxes[:,:,2], gtboxes[:,:,2]) - np.maximum(dtboxes[:,:,0], gtboxes[:,:,0])
    ih = np.minimum(dtboxes[:,:,3], gtboxes[:,:,3]) - np.maximum(dtboxes[:,:,1], gtboxes[:,:,1])
    inter = np.maximum(0, iw) * np.maximum(0, ih)

    dtarea = (dtboxes[:,:,2] - dtboxes[:,:,0]) * (dtboxes[:,:,3] - dtboxes[:,:,1])
    gtarea = (gtboxes[:,:,2] - gtboxes[:,:,0]) * (gtboxes[:,:,3] - gtboxes[:,:,1])
    ious = inter / (dtarea + gtarea - inter + eps)
    return ious
def compute_lap(dtboxes, gtboxes, thr):

    eps = 1e-7
    n, k = dtboxes.shape[0], gtboxes.shape[0]
    if k + n < 2:
        m, n = np.array([]), np.array([])
        return m, n
        
    overlaps = compute_iou_matrix(dtboxes, gtboxes)

    if n < 2:
        cols = np.argmax(overlaps, axis = 1)
        rows = np.array([0])
        m, n = (rows, cols) if thr - overlaps[rows, cols] < eps else (np.array([]), np.array([]))
        return m, n

    if k < 2:
        
        rows = np.argmax(overlaps, axis = 0)
        cols = np.array([0])
        m,n = (rows, cols) if thr - overlaps[rows, cols] < eps else (np.array([]), np.array([]))
        return m, n
            
    ious = overlaps * (overlaps >= thr)
        
    matches = minimumWeightMatching(-ious)
    m, n = np.array([i for i, _ in matches]).astype(np.int32), np.array([i for _, i in matches]).astype(np.int32)
    indice = np.where(overlaps[m, n] < thr)[0]

    if indice.size >= m.size:
        m, n = np.array([]), np.array([])
    else:
        index = np.array(list(set(np.arange(m.size)) - set(indice))).astype(np.int)
        m, n = m[index], n[index]
    
    return m, n
def compute_JC(detection:np.ndarray, gt:np.ndarray, iou_thresh:np.ndarray):

    rows, cols = compute_lap(detection, gt, iou_thresh)
    return [(i, j) for i, j in zip(rows, cols)]

def worker(result_queue, pred_list, gt_list, score_thr, bm_thr):

    result_list = []
    total, eps = len(pred_list), 1e-6
    for i in range(total):
        record = pred_list[i]
        GT = gt_list[i]
        assert GT['ID'] == record['ID'] 
        ID = record['ID']
        if len(record['dtboxes']) < 1:
            # result_list.append(None)
            continue

        # GT = list(filter(lambda rb:rb['ID'] == ID, gt))
        # if len(GT) < 1:
        #     result_queue.put_nowait(None)
        #     continue
        
        # GT = GT[0]
        if 'height' in record and 'width' in record:
            height, width = record['height'], record['width']
        else:
            height, width = GT['height'], GT['width']
        flags = np.array([is_ignore(rb) for rb in GT['gtboxes']])
        rows = np.where(~flags)[0]
        ignores = np.where(flags)[0]

        gtboxes = np.vstack([GT['gtboxes'][j]['fbox'] for j in rows])
        gtboxes = recover_func(gtboxes)
        gtboxes = clip_boundary(gtboxes, height, width)

        if ignores.size:
            ignores = np.vstack([GT['gtboxes'][j]['fbox'] for j in ignores])
            ignores = recover_func(ignores)
            ignores = clip_boundary(ignores, height, width)

        dtboxes = np.vstack([np.hstack([rb['box'], rb['score']]) for rb in record['dtboxes']])
        dtboxes = recover_func(dtboxes)
        dtboxes = clip_boundary(dtboxes, height, width)
        rows = np.where(dtboxes[:,-1]> score_thr)[0]
        dtboxes = dtboxes[rows,...]

        matches = compute_JC(dtboxes, gtboxes, bm_thr)
        dt_ign, gt_ign = 0, 0

        if ignores.size:
            indices = np.array([j for (j,_) in matches])
            dt_ign = get_ignores(indices, dtboxes, ignores, bm_thr)
            indices = np.array([j for (_,j) in matches])
            gt_ign = get_ignores(indices, gtboxes, ignores, bm_thr)

        k = len(matches)
        m = gtboxes.shape[0] - gt_ign
        n = dtboxes.shape[0] - dt_ign

        ratio = k / (m + n -k + eps)
        recall = k / (m + eps)
        cover = k / (n + eps)
        noise = 1 - cover

        result_dict = dict(ID = ID, ratio = ratio, recall = recall , noise = noise ,
            cover = cover, k= k ,n = n, m = m)
        result_list.append(result_dict)
    ji = np.sum([rb['ratio'] for rb in result_list]) / 4370
    result_queue.put_nowait(ji)

def common_process(func, data, nr_procs, *args):

    total = len(data)
    stride = math.ceil(total / nr_procs)
    result_queue = Queue(10000)
    results, procs = [], []
    tqdm.monitor_interval = 0
    pbar = tqdm(total = total, leave = False, ascii = True)
    for i in range(nr_procs):
        start = i*stride
        end = np.min([start+stride,total])
        sample_data = data[start:end]
        # func(result_queue, sample_data, *args)
        p = Process(target= func,args=(result_queue, sample_data, *args))
        p.start()
        procs.append(p)

    for i in range(total):

        t = result_queue.get()
        if t is None:
            pbar.update(1)
            continue
        results.append(t)
        pbar.update()
    for p in procs:
        p.join()
    return results

def process_jaccard_computation(pred_list, gt_list, bm_thr):

    result_queue = Queue(10000)
    results, procs = [], []
    total = 10
    pbar = tqdm(total = total, leave = False, ascii = True)
    for i in range(total):
        p = Process(target= worker,args=(result_queue, pred_list, gt_list, i*0.1, bm_thr))
        p.start()
        procs.append(p)

    for i in range(total):
        t = result_queue.get()
        if t is None:
            pbar.update(1)
            continue
        results.append(t)
        pbar.update()
    for p in procs:
        p.join()
    return max(results)
def computeJaccard(gt_list, dt_list, conf_range, bm_thr=0.5):
    # from demo import load_func, common_process, worker
    gt_list.sort(key=lambda x:x['ID'])
    dt_list.sort(key=lambda x:x['ID'])
    ji = 0.
    ji= process_jaccard_computation(dt_list, gt_list, bm_thr)
    # for i in conf_range:
    
    #     results = common_process(worker, dt_list, 4, gt_list, i * 0.1, 0.5)
    #     ji = max(ji, np.sum([rb['ratio'] for rb in results]) / 4370)
    #     print('compute jaccard index', i, ji)
    return ji

class Image(object):
    def __init__(self, mode):
        self.ID = None
        self._width = None
        self._height = None
        self.dtboxes = None
        self.gtboxes = None
        self.eval_mode = mode

        self._ignNum = None
        self._gtNum = None
        self._dtNum = None
    def load_cocojson(self,  img_item, annots, is_gt =True):
        """
        :meth: read the object from a dict
        """
        if self.ID is None:
            self.ID = img_item['id']
        if self._width is None:
            self._width = img_item["width"]
        if self._height is None:
            self._height = img_item["height"]
        
        # body_bbox, head_bbox = self.load_gt_boxes(recoannotrd, 'gtboxes', class_names)
        boxes = np.array([annot['bbox']  for annot in annots])
      
        if len(boxes)>0:
            #xywh => xyxy
            boxes[:,2:4] = boxes[:,:2] + boxes[:,2:4]    
        else:
            boxes = np.zeros((0,4))
        if is_gt:
            boxes = np.concatenate([boxes, np.ones((len(boxes), 1))], axis=-1)
            self._gtNum = len(boxes)
            self.gtboxes = boxes
            self._ignNum = (boxes[:, -1] == -1).sum()
        else:
            #if score in annot, concate it with boxes
            if len(annots) > 0 and 'score' in annots[0].keys():
                scores = np.array([annot['score'] for annot in annots])[:, None]
            else:
                scores = np.ones((len(boxes), 1))
            boxes = np.concatenate([boxes, scores], axis=-1)
            
            self._dtNum = len(boxes)
            self.dtboxes = boxes       
    def load(self, record, body_key, head_key, class_names, gtflag, visible_flag=True):
        """
        :meth: read the object from a dict
        """
        if "ID" in record and self.ID is None:
            self.ID = record['ID']
        if "width" in record and self._width is None:
            self._width = record["width"]
        if "height" in record and self._height is None:
            self._height = record["height"]
        if gtflag:
            self._gtNum = len(record["gtboxes"])
            body_bbox, head_bbox = self.load_gt_boxes(record, 'gtboxes', class_names, visible_flag)
            if self.eval_mode == 0:
                self.gtboxes = body_bbox
                self._ignNum = (body_bbox[:, -1] == -1).sum()
            elif self.eval_mode == 1:
                self.gtboxes = head_bbox
                self._ignNum = (head_bbox[:, -1] == -1).sum()
            elif self.eval_mode == 2:
                gt_tag = np.array(
                    [body_bbox[i, -1] != -1 and head_bbox[i, -1] != -1
                     for i in range(len(body_bbox))]
                )
                self._ignNum = (gt_tag == 0).sum()
                self.gtboxes = np.hstack(
                    (body_bbox[:, :-1], head_bbox[:, :-1], gt_tag.reshape(-1, 1))
                )
            else:
                raise Exception('Unknown evaluation mode!')
        if not gtflag:
            self._dtNum = len(record["dtboxes"])
            if self.eval_mode == 0:
                self.dtboxes = self.load_det_boxes(record, 'dtboxes', body_key, 'score')
            elif self.eval_mode == 1:
                self.dtboxes = self.load_det_boxes(record, 'dtboxes', head_key, 'score')
            elif self.eval_mode == 2:
                body_dtboxes = self.load_det_boxes(record, 'dtboxes', body_key)
                head_dtboxes = self.load_det_boxes(record, 'dtboxes', head_key, 'score')
                self.dtboxes = np.hstack((body_dtboxes, head_dtboxes))
            else:
                raise Exception('Unknown evaluation mode!')

    def compare_caltech(self, thres):
        """
        :meth: match the detection results with the groundtruth by Caltech matching strategy
        :param thres: iou threshold
        :type thres: float
        :return: a list of tuples (dtbox, imageID), in the descending sort of dtbox.score
        """
        if self.dtboxes is None or self.gtboxes is None:
            return list()

        dtboxes = self.dtboxes if self.dtboxes is not None else list()
        gtboxes = self.gtboxes if self.gtboxes is not None else list()

        dtboxes = np.array(sorted(dtboxes, key=lambda x: x[-1], reverse=True))
        gtboxes = np.array(sorted(gtboxes, key=lambda x: x[-1], reverse=True))
        if len(dtboxes):
            overlap_iou = self.box_overlap_opr(dtboxes, gtboxes[gtboxes[:, -1] > 0], True)
            overlap_ioa = self.box_overlap_opr(dtboxes, gtboxes[gtboxes[:, -1] <= 0], False)
            ign = np.any(overlap_ioa > thres, 1)
            pos = np.any(overlap_iou > thres, 1)
        else:
            return list()

        scorelist = list()
        for i, dt in enumerate(dtboxes):
            maxpos = np.argmax(overlap_iou[i])
            if overlap_iou[i, maxpos] > thres:
                overlap_iou[:, maxpos] = 0
                scorelist.append((dt, 1, self.ID, pos[i]))
            elif not ign[i]:
                scorelist.append((dt, 0, self.ID, pos[i]))
        return scorelist

    def compare_caltech_union(self, thres):
        """
        :meth: match the detection results with the groundtruth by Caltech matching strategy
        :param thres: iou threshold
        :type thres: float
        :return: a list of tuples (dtbox, imageID), in the descending sort of dtbox.score
        """
        dtboxes = self.dtboxes if self.dtboxes is not None else list()
        gtboxes = self.gtboxes if self.gtboxes is not None else list()
        if len(dtboxes) == 0:
            return list()
        dt_matched = np.zeros(dtboxes.shape[0])
        gt_matched = np.zeros(gtboxes.shape[0])

        dtboxes = np.array(sorted(dtboxes, key=lambda x: x[-1], reverse=True))
        gtboxes = np.array(sorted(gtboxes, key=lambda x: x[-1], reverse=True))
        dt_body_boxes = np.hstack((dtboxes[:, :4], dtboxes[:, -1][:, None]))
        dt_head_boxes = dtboxes[:, 4:8]
        gt_body_boxes = np.hstack((gtboxes[:, :4], gtboxes[:, -1][:, None]))
        gt_head_boxes = gtboxes[:, 4:8]
        overlap_iou = self.box_overlap_opr(dt_body_boxes, gt_body_boxes, True)
        overlap_head = self.box_overlap_opr(dt_head_boxes, gt_head_boxes, True)
        overlap_ioa = self.box_overlap_opr(dt_body_boxes, gt_body_boxes, False)

        scorelist = list()
        for i, dt in enumerate(dtboxes):
            maxpos = -1
            maxiou = thres
            for j, gt in enumerate(gtboxes):
                if gt_matched[j] == 1:
                    continue
                if gt[-1] > 0:
                    o_body = overlap_iou[i][j]
                    o_head = overlap_head[i][j]
                    if o_body > maxiou and o_head > maxiou:
                        maxiou = o_body
                        maxpos = j
                else:
                    if maxpos >= 0:
                        break
                    else:
                        o_body = overlap_ioa[i][j]
                        if o_body > thres:
                            maxiou = o_body
                            maxpos = j
            if maxpos >= 0:
                if gtboxes[maxpos, -1] > 0:
                    gt_matched[maxpos] = 1
                    dt_matched[i] = 1
                    scorelist.append((dt, 1, self.ID))
                else:
                    dt_matched[i] = -1
            else:
                dt_matched[i] = 0
                scorelist.append((dt, 0, self.ID))
        return scorelist

    def box_overlap_opr(self, dboxes: np.ndarray, gboxes: np.ndarray, if_iou) -> np.ndarray:
        eps = 1e-6
        assert dboxes.shape[-1] >= 4 and gboxes.shape[-1] >= 4
        N, K = dboxes.shape[0], gboxes.shape[0]
        dtboxes = np.tile(np.expand_dims(dboxes, axis=1), (1, K, 1))
        gtboxes = np.tile(np.expand_dims(gboxes, axis=0), (N, 1, 1))

        iw = (np.minimum(dtboxes[:, :, 2], gtboxes[:, :, 2])
              - np.maximum(dtboxes[:, :, 0], gtboxes[:, :, 0]))
        ih = (np.minimum(dtboxes[:, :, 3], gtboxes[:, :, 3])
              - np.maximum(dtboxes[:, :, 1], gtboxes[:, :, 1]))
        inter = np.maximum(0, iw) * np.maximum(0, ih)

        dtarea = (dtboxes[:, :, 2] - dtboxes[:, :, 0]) * (dtboxes[:, :, 3] - dtboxes[:, :, 1])
        if if_iou:
            gtarea = (gtboxes[:, :, 2] - gtboxes[:, :, 0]) * (gtboxes[:, :, 3] - gtboxes[:, :, 1])
            ious = inter / (dtarea + gtarea - inter + eps)
        else:
            ious = inter / (dtarea + eps)
        return ious

    def clip_all_boader(self):

        def _clip_boundary(boxes, height, width):
            assert boxes.shape[-1] >= 4
            boxes[:, 0] = np.minimum(np.maximum(boxes[:, 0], 0), width - 1)
            boxes[:, 1] = np.minimum(np.maximum(boxes[:, 1], 0), height - 1)
            boxes[:, 2] = np.maximum(np.minimum(boxes[:, 2], width), 0)
            boxes[:, 3] = np.maximum(np.minimum(boxes[:, 3], height), 0)
            return boxes

        assert self.dtboxes.shape[-1] >= 4
        assert self.gtboxes.shape[-1] >= 4
        assert self._width is not None and self._height is not None
        if self.eval_mode == 2:
            self.dtboxes[:, :4] = _clip_boundary(self.dtboxes[:, :4], self._height, self._width)
            self.gtboxes[:, :4] = _clip_boundary(self.gtboxes[:, :4], self._height, self._width)
            self.dtboxes[:, 4:8] = _clip_boundary(self.dtboxes[:, 4:8], self._height, self._width)
            self.gtboxes[:, 4:8] = _clip_boundary(self.gtboxes[:, 4:8], self._height, self._width)
        else:
            self.dtboxes = _clip_boundary(self.dtboxes, self._height, self._width)
            self.gtboxes = _clip_boundary(self.gtboxes, self._height, self._width)

    def load_gt_boxes(self, dict_input, key_name, class_names, visible_flag):
        assert key_name in dict_input
        if len(dict_input[key_name]) < 1:
            return np.empty([0, 5])
        head_bbox = []
        body_bbox = []
        for rb in dict_input[key_name]:
            if rb['tag'] in class_names:
                body_tag = class_names.index(rb['tag'])
                head_tag = 1
            else:
                body_tag = -1
                head_tag = -1
            if 'extra' in rb:
                if 'ignore' in rb['extra']:
                    if rb['extra']['ignore'] != 0:
                        body_tag = -1
                        head_tag = -1
            if 'head_attr' in rb:
                if 'ignore' in rb['head_attr']:
                    if rb['head_attr']['ignore'] != 0:
                        head_tag = -1
            # head_bbox.append(np.hstack((rb['hbox'], head_tag)))
            if visible_flag:
                body_bbox.append((*rb['vbox'][0], body_tag))
            else:
                body_bbox.append((*rb['fbox'], body_tag))
        # head_bbox = np.array(head_bbox)
        # head_bbox[:, 2:4] += head_bbox[:, :2]
        body_bbox = np.array(body_bbox)
        
        body_bbox[:, 2:4] += body_bbox[:, :2]
        return body_bbox, head_bbox

    def load_det_boxes(self, dict_input, key_name, key_box, key_score=None, key_tag=None):
        assert key_name in dict_input
        if len(dict_input[key_name]) < 1:
            return np.empty([0, 5])
        else:
            assert key_box in dict_input[key_name][0]
            if key_score:
                assert key_score in dict_input[key_name][0]
            if key_tag:
                assert key_tag in dict_input[key_name][0]
        if key_score:
            if key_tag:
                bboxes = np.vstack(
                    [
                        np.hstack(
                            (rb[key_box], rb[key_score], rb[key_tag])
                        ) for rb in dict_input[key_name]
                    ]
                )
            else:
                bboxes = np.array([(*rb[key_box], rb[key_score]) for rb in dict_input[key_name]])
        else:
            if key_tag:
                bboxes = np.vstack(
                    [np.hstack((rb[key_box], rb[key_tag])) for rb in dict_input[key_name]]
                )
            else:
                bboxes = np.vstack([rb[key_box] for rb in dict_input[key_name]])
        bboxes[:, 2:4] += bboxes[:, :2]
        return bboxes

    def compare_voc(self, thres):
        """
        :meth: match the detection results with the groundtruth by VOC matching strategy
        :param thres: iou threshold
        :type thres: float
        :return: a list of tuples (dtbox, imageID), in the descending sort of dtbox.score
        """
        if self.dtboxes is None:
            return list()
        dtboxes = self.dtboxes
        gtboxes = self.gtboxes if self.gtboxes is not None else list()
        dtboxes.sort(key=lambda x: x.score, reverse=True)
        gtboxes.sort(key=lambda x: x.ign)

        scorelist = list()
        for i, dt in enumerate(dtboxes):
            maxpos = -1
            maxiou = thres

            for j, gt in enumerate(gtboxes):
                overlap = dt.iou(gt)
                if overlap > maxiou:
                    maxiou = overlap
                    maxpos = j

            if maxpos >= 0:
                if gtboxes[maxpos].ign == 0:
                    gtboxes[maxpos].matched = 1
                    dtboxes[i].matched = 1
                    scorelist.append((dt, self.ID))
                else:
                    dtboxes[i].matched = -1
            else:
                dtboxes[i].matched = 0
                scorelist.append((dt, self.ID))
        return scorelist

