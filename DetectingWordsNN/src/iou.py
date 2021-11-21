import numpy as np


def compute_iou(ra, rb):
    """Вычисляет площадь пересечения границ слова которые предстазала сетка с границами которые были установлены при
    обучении площадь пересечения этих границ делит на площадь объединения.
    для одного слова это IOU"""
    if ra.xmax < rb.xmin or rb.xmax < ra.xmin or ra.ymax < rb.ymin or rb.ymax < ra.ymin:
        return 0

    l = max(ra.xmin, rb.xmin)
    r = min(ra.xmax, rb.xmax)
    t = max(ra.ymin, rb.ymin)
    b = min(ra.ymax, rb.ymax)

    intersection = (r - l) * (b - t)
    union = ra.area() + rb.area() - intersection

    iou = intersection / union
    return iou


def compute_dist_mat(aabbs):
    """Находим расстояние между всеми словами попарно"""
    num_aabbs = len(aabbs)

    dists = np.zeros((num_aabbs, num_aabbs))
    for i in range(num_aabbs):
        for j in range(num_aabbs):
            if j > i:
                break

            dists[i, j] = dists[j, i] = 1 - compute_iou(aabbs[i], aabbs[j])

    return dists


def compute_dist_mat_2(aabbs1, aabbs2):
    """Находим расстояние попарно между двумя списками слов"""
    num_aabbs1 = len(aabbs1)
    num_aabbs2 = len(aabbs2)

    dists = np.zeros((num_aabbs1, num_aabbs2))
    for i in range(num_aabbs1):
        for j in range(num_aabbs2):
            dists[i, j] = 1 - compute_iou(aabbs1[i], aabbs2[j])

    return dists
