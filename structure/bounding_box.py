import numpy as np
FLIP_LR = 1
FLIP_UD = 0


def BoxListCat(boxlist):
    size = boxlist[0].size
    #print(size)
    bboxes = []
    labels = []
    scores = []
    for b in boxlist:
        assert size == b.size,("box must in the same scale")
        bboxes.append(b.box)
        labels.extend(b.get_field('labels'))
        scores.extend(b.get_field('scores'))

    box = np.concatenate(bboxes,axis=0)
    scores = np.asarray(scores)
    bboxes = BoxList(box, size)
    bboxes.add_field('scores', scores)
    bboxes.add_field('labels', labels)
    #print(bboxes.size)
    return bboxes

class BoxList(object):

    def __init__(self,box,size,mode="xyxy"):
        self.box = np.asarray(box)
        self.size = size
        self.mode = mode
        self.extra_fields = {}
        assert self.mode == "xyxy"

    def copy(self):
        new = BoxList(self.box.copy(),self.size)
        for k, v in self.extra_fields.items():
            new.extra_fields[k] = v.copy()
        return new


    def add_field(self, field, field_data):
        self.extra_fields[field] = field_data

    def get_field(self, field):
        return self.extra_fields[field]

    def has_field(self, field):
        return field in self.extra_fields

    def fields(self):
        return list(self.extra_fields.keys())

    def _copy_extra_fields(self, bbox):
        for k, v in bbox.extra_fields.items():
            self.extra_fields[k] = v

    def resize(self,r_size):
        w,h = self.size
        n_w, n_h = r_size

        self.box[:,0] = self.box[:,0] / w * n_w
        self.box[:,1] = self.box[:,1] / h * n_h
        self.box[:,2] = self.box[:,2] / w * n_w
        self.box[:,3] = self.box[:,3] / h * n_h

        self.size = r_size
        return #self.box

    def crop(self,box):
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1

        self.box[:,0] = np.clip(self.box[:,0] - x1,1,w)
        self.box[:,1] = np.clip(self.box[:,1] - y1,1,h)
        self.box[:,2] = np.clip(self.box[:,2] - x1,1,w)
        self.box[:,3] = np.clip(self.box[:,3] - y1,1,h)
        self.size = (w,h)

        return

    def flip(self,ops):
        w, h = self.size
        if ops == FLIP_LR:
            bw = self.box[:,2] - self.box[:,0]
            self.box[:,0] = np.clip(w-(self.box[:,0]+bw),1,w)
            self.box[:,2] = np.clip(w-(self.box[:,2]-bw),1,w)
            self.box[:,1] = np.clip(self.box[:,1],1,h)
            self.box[:,3] = np.clip(self.box[:,3],1,h)
        if ops == FLIP_UD:
            bh = self.box[:,3] - self.box[:,1]
            self.box[:,0] = np.clip(self.box[:,0],1,w)
            self.box[:,2] = np.clip(self.box[:,2],1,w)
            self.box[:,1] = np.clip(h-(self.box[:,1]+bh),1,h)
            self.box[:,3] = np.clip(h-(self.box[:,3]-bh),1,h)
        else:
            ValueError("Only support 0,1")
        return

    def warpAffine(self,matrix,size=None):
        if size:
            self.size = size
        p1_list = np.concatenate([self.box[:, :2], np.ones((self.box.shape[0], 1))], axis=1)
        p2_list = np.concatenate([self.box[:, 2:4], np.ones((self.box.shape[0], 1))], axis=1)

        box = []
        for p1, p2 in zip(p1_list, p2_list):
            new_pt1 = np.dot(matrix, p1.T)
            new_pt2 = np.dot(matrix, p2.T)

            new_pt1[0] = min(max(1,new_pt1[0]),self.size[0])
            new_pt1[1] = min(max(1,new_pt1[1]),self.size[1])
            new_pt2[0] = min(max(1,new_pt2[0]),self.size[0])
            new_pt2[1] = min(max(1,new_pt2[1]),self.size[1])

            box += [[new_pt1[0], new_pt1[1], new_pt2[0], new_pt2[1]]]

        self.box = np.asarray(box)

    def rot90(self,time):

        time = time % 4
        for i in range(time):
            w, h = self.size
            flipped = self.box.copy()
            flipped[:, 0] = self.box[:, 1]
            flipped[:, 2] = self.box[:, 3]
            flipped[:, 1] = w - self.box[:, 2]
            flipped[:, 3] = w - self.box[:, 0]
            self.box = flipped
            self.size = h, w

        w, h = self.size
        self.box[:,0] = np.clip(self.box[:,0],1,w)
        self.box[:,1] = np.clip(self.box[:,1],1,h)
        self.box[:,2] = np.clip(self.box[:,2],1,w)
        self.box[:,3] = np.clip(self.box[:,3],1,h)

        return

    def area(self):
        box = self.box
        assert box.shape[0] > 0,box
        assert box.shape[1] == 4,box
        if self.mode == "xyxy":
            TO_REMOVE = 0
            area = (box[:, 2] - box[:, 0] + TO_REMOVE) * (box[:, 3] - box[:, 1] + TO_REMOVE)
        else:
            raise RuntimeError("Should not be here")

        return area

    def __len__(self):
        return self.box.shape[0]

# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def boxlist_iou(boxlist1, boxlist2):
    """Compute the intersection over union of two set of boxes.
    The box order must be (xmin, ymin, xmax, ymax).
    Arguments:
      box1: (BoxList) bounding boxes, sized [N,4].
      box2: (BoxList) bounding boxes, sized [M,4].
    Returns:
      (tensor) iou, sized [N,M].
    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """
    if boxlist1.size != boxlist2.size:
        raise RuntimeError(
                "boxlists should have same image size, got {}, {}".format(boxlist1, boxlist2))

    N = len(boxlist1)
    M = len(boxlist2)

    area1 = boxlist1.area()
    area2 = boxlist2.area()

    box1, box2 = boxlist1.box, boxlist2.box

    lt = np.maximum(box1[:,None, :2], box2[:, :2])  # [N,M,2]
    rb = np.minimum(box1[:,None, 2:], box2[:, 2:])  # [N,M,2]

    TO_REMOVE = 1

    wh = np.clip(rb - lt + TO_REMOVE,0,None)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:,:, 1]  # [N,M]

    iou = inter / (area1[:,None] + area2 - inter)
    return iou




from torch import nn

def pool_nms(heat, kernel=3):
    heat = heat.sigmoid()

    pad = (kernel - 1) // 2
    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def pool_nms_no_sigmoid(heat, kernel=3):

    pad = (kernel - 1) // 2
    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

