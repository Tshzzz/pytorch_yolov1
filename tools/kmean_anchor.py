import numpy as np


def iou(box, clusters):
    """
    Calculates the Intersection over Union (IoU) between a box and k clusters.
    :param box: tuple or array, shifted to the origin (i. e. width and height)
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: numpy array of shape (k, 0) where k is the number of clusters
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        raise ValueError("Box has no area")

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = intersection / (box_area + cluster_area - intersection)

    return iou_


def avg_iou(boxes, clusters):
    """
    Calculates the average Intersection over Union (IoU) between a numpy array of boxes and k clusters.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: average IoU as a single float
    """
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])


def translate_boxes(boxes):
    """
    Translates all the boxes to the origin.
    :param boxes: numpy array of shape (r, 4)
    :return: numpy array of shape (r, 2)
    """
    new_boxes = boxes.copy()
    for row in range(new_boxes.shape[0]):
        new_boxes[row][2] = np.abs(new_boxes[row][2] - new_boxes[row][0])
        new_boxes[row][3] = np.abs(new_boxes[row][3] - new_boxes[row][1])
    return np.delete(new_boxes, [0, 1], axis=1)


def kmeans(boxes, k, dist=np.median):
    """
    Calculates k-means clustering with the Intersection over Union (IoU) metric.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param k: number of clusters
    :param dist: distance function
    :return: numpy array of shape (k, 2)
    """
    rows = boxes.shape[0]

    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))

    np.random.seed(666)

    # the Forgy method will fail if the whole array contains the same rows
    clusters = boxes[np.random.choice(rows, k, replace=False)]

    while True:
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)

        nearest_clusters = np.argmin(distances, axis=1)

        if (last_clusters == nearest_clusters).all():
            break

        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)

        last_clusters = nearest_clusters

    return clusters


def norm_bb(b, size):
    x = b[:, 0:1]
    y = b[:, 1:2]

    dw = 1. / size[0]
    dh = 1. / size[1]

    x = (x * dw)  # .clip(0.01, 0.99)
    y = (y * dh)  # .clip(0.01, 0.99)
    w = ((b[:, 2:3] - b[:, 0:1]) * dw)  # .clip(0.01, 0.99)
    h = ((b[:, 3:4] - b[:, 1:2]) * dh)  # .clip(0.01, 0.99)
    # b[:, 4:5] = 1

    return np.concatenate((x, y, w, h, b[:, 4:5]), axis=1)


if __name__ == '__main__':
    import cv2
    import tqdm
    list_file = '/home/tshzzz/Documents/mygithub/pytorch_yolov1/train_list/train.txt'
    with open(list_file) as f:
        lines = f.readlines()

    image_path = []
    label_path = []
    for line in lines:
        splited = line.strip().split(' ')
        label_path.append(splited[1])
        image_path.append(splited[0])

    data = []
    for img,label in  tqdm.tqdm(zip(image_path,label_path)):
        a = cv2.imread(img)
        bs = np.loadtxt(label,delimiter=',').reshape(-1,5).astype(np.float16)
        bs = norm_bb(bs, (a.shape[1], a.shape[0]))
        for wh in bs:
            data.append([wh[2],wh[3]])


    data = np.array(data)
    print(data.shape)
    print(np.mean(data[:, 0]))
    print(np.mean(data[:, 1]))

    out = kmeans(data, k=3)
    print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))
    print("Boxes:\n {}".format(out))

    ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
    print("Ratios:\n {}".format(sorted(ratios)))
