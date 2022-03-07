import uuid
import matplotlib.pyplot as plotlib
import numpy
import collections


ps = []


def vis_single(img, p, midx):
    prefix = str(uuid.uuid4())
    img = (img - img.min()) / (img.max() - img.min())
    plotlib.figure()
    plotlib.imshow(img.transpose((1, 2, 0)))
    plotlib.scatter(p[:, 0], p[:, 1])
    s = collections.defaultdict(int)
    for i in range(0, 10, 2):
        max_idx = midx[i]  # D' x S, 0 ~ K
        group_idx = midx[i + 1]  # S x K
        S, K = group_idx.shape
        max_idx_revisit = group_idx[numpy.arange(S).reshape(1, -1), max_idx]  # D' x S
        for j in range(len(max_idx_revisit)):
            for k in range(S):
                s[k, max_idx_revisit[j, k]] += 1
                s[max_idx_revisit[j, k], k] += 1
    for i in range(S):
        s[i, i] = 0
    maxed = max(s.values())
    for i in range(S):
        for j in range(S):
            plotlib.plot([p[i, 0], p[j, 0]], [p[i, 1], p[j, 1]], alpha=s[i, j] / maxed)
    plotlib.savefig("aggr/" + prefix + ".png", dpi=240)


def visualize(img, p, ns, midx):
    batch = len(img)
    for i in range(batch):
        vis_single(img[i], p[i][:ns[i]], [m[i] for m in midx])
