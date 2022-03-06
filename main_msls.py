import matplotlib.pyplot as plotlib
import cv2
import torch
import torchvision.transforms as transforms
import numpy
import os
from src.dataset.data_loader import GMDataset, get_dataloader
from src.utils.model_sl import load_model
from src.utils.config import cfg


IMG_ROOT = "../msls/SuperPointPretrainedNetwork-master/val"
DET_PATH = "../msls/SuperPointPretrainedNetwork-master/detection.npz"
detection = numpy.load(DET_PATH)


def data_prepare(key, renorm=True):
    trans = transforms.Compose([
        # transforms.RandomApply([
        #     transforms.Resize([random.randint(64, 224)] * 2),
        #     transforms.Resize([256, 256])
        # ]),
        transforms.ToTensor(),
        (lambda x: x) if not renorm else transforms.Normalize(cfg.NORM_MEANS, cfg.NORM_STD)
    ])
    pts = detection[key].T  # N x [x, y, confidence]
    top20 = pts[pts[:, 2].argsort()[-50:]]
    img = plotlib.imread(IMG_ROOT + "/" + key)  # hwc, rgb
    xy = top20[:, :2] / numpy.array([640, 480]) * 256
    img = cv2.resize(img, (256, 256))
    return xy.astype('float32'), trans(img)


@torch.no_grad()
def main():
    from src.utils.parse_args import parse_args
    parse_args('Deep learning of graph matching visualization code.')
    
    import importlib
    from src.utils.config import cfg_from_file
    cfg_from_file('experiments/igm.yaml')

    mod = importlib.import_module(cfg.MODULE)
    Net = mod.Net

    model = Net().cuda()
    load_model(model, 'output/igm_voc/params/params_9999.pt')
    model.eval()

    for kq in detection.keys():
        if '_q' not in kq:
            continue
        kd = kq.replace("_q", "_d")
        imgs, ps, ns = [], [], []
        imgs.append(data_prepare(kq)[1].cuda().unsqueeze(0))
        imgs.append(data_prepare(kd)[1].cuda().unsqueeze(0))
        ps.append(torch.tensor(data_prepare(kq)[0]).cuda().unsqueeze(0))
        ns.append(torch.tensor(len(ps[-1][0])).cuda().unsqueeze(0))
        ps.append(torch.tensor(data_prepare(kd)[0]).cuda().unsqueeze(0))
        ns.append(torch.tensor(len(ps[-1][0])).cuda().unsqueeze(0))
        result = model({"images": imgs, "Ps": ps, "ns": ns})
        match = result['perm_mat'].detach().cpu().numpy()[0]
        img = torch.cat([data_prepare(kq, 0)[1], data_prepare(kd, 0)[1]], -1).permute(1, 2, 0).cpu().numpy()
        
        plotlib.figure()
        plotlib.imshow(img)
        psma, psmb = [], []
        for i in range(match.shape[0]):
            for j in range(match.shape[1]):
                if match[i, j] == 1:
                    psma.append(ps[0][0][i].cpu().numpy())
                    psmb.append(ps[1][0][j].cpu().numpy())
                    # plotlib.plot([ps[0][0][i, 0].item(), ps[1][0][j, 0].item() + 256],
                    #              [ps[0][0][i, 1].item(), ps[1][0][j, 1].item()])
        _, mask = cv2.findHomography(numpy.array(psma), numpy.array(psmb), cv2.RANSAC)
        print(kq, mask.sum())
        for a, b, m in zip(psma, psmb, mask):
            if m:
                plotlib.plot([a[0], b[0] + 256], [a[1], b[1]])
        plotlib.scatter(ps[0][0][:, 0].cpu(), ps[0][0][:, 1].cpu(), c='white', s=5)
        plotlib.scatter(ps[1][0][:, 0].cpu() + 256, ps[1][0][:, 1].cpu(), c='white', s=5)
        os.makedirs("match", exist_ok=True)
        plotlib.savefig("match/" + kq, dpi=300, quality=95)

if __name__ == '__main__':
    main()
