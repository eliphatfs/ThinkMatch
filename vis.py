import torch
import numpy as np
from pathlib import Path

from src.dataset.data_loader import GMDataset, get_dataloader
from src.utils.model_sl import load_model
from src.parallel import DataParallel
from src.lap_solvers.hungarian import hungarian
from src.utils.data_to_cuda import data_to_cuda
import matplotlib
import pandas as pd
#import seaborn as sb
try:
    import _tkinter
except ImportError:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import math

plt.rcParams["font.family"] = "serif"

from src.utils.config import cfg

font_dict = {'family' : 'sans-serif',
'weight' : 'bold',
'size'   : 15,
'color' : 'white',
}

def vertical_subplt(a,b,c):
    plt.subplot(b, a, (c // b) + c % b * a)

def draw_heatmap(data, index, columns):
    df = pd.DataFrame(data=data, index=index, columns=columns)
    fig, ax = plt.subplots(figsize=(9, 9))
    sb.heatmap(df, cmap="Blues", annot=True, vmin=0., vmax=1., cbar=False, square=True)
    title='Confusion matrix'
    plt.title(title, y=-0.2, fontsize=25)
    plt.savefig('figs/debug.pdf')
    plt.show()
    return 

def heatmap(data, row_labels, col_labels, ax=None, if_cbar=False,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        fig = plt.figure()
        ax = fig.add_subplot()
        #ax = plt.gca()
    fontsize=8
    # Plot the heatmap
    im = ax.imshow(data, cmap='Oranges', **kwargs)

    # Create colorbar
    if if_cbar:
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels, fontsize=fontsize, fontweight='bold')
    ax.set_yticklabels(row_labels, fontsize=fontsize, fontweight='bold')

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                    labeltop=True, labelbottom=False, length=0)
    #ax.tick_params(top=False, bottom=True,
    #              labeltop=False, labelbottom=True, length=0)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=35, ha="left",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    #ax.spines[:].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['top'].set_visible(True)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="#B0B0B0", linestyle='-', linewidth=1)
    # ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(which="minor", top=False, left=False)
    plt.subplots_adjust(top=0.83)
    plt.tight_layout()

    if if_cbar:
        return im, cbar
    else:
        return im, None

def annotate_heatmap(im, title, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """
    fontsize=35
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center",
              fontsize=fontsize)
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            # if title == 'semantic aware matrix' or 'semantic aware matrix(gt)':
            #     data_grid = data[i, j]
            #     if data_grid == -1.:
            #         color='#7F2704'
            #     elif data_grid == 0.:
            #         color='#FFF5EB'
            #     elif data_grid == 1.:
            #         color='#3C0477'
            # else:
            if title == 'semantic aware matrix' or title == 'semantic aware matrix(gt)':
                if data[i, j] == 2.:
                    data_grid = -1.
                    color = 'white'
                else:
                    data_grid = data[i, j]
                    color = textcolors[int(im.norm(data[i, j]) > threshold)]
            elif title == 'SAR matrix':
                data_grid = data[i, j]                
                if i != j:
                    data_grid *= -1
                    color='black'
                else:
                    if i == 5:
                        color = 'black'
                    else:
                        color='white'
                if data_grid == -0.:
                    data_grid = 0.
                # color = textcolors[int(im.norm(data[i, j]) > threshold)]                
            else:
                if data[i, j] == -2.:
                    data_grid = 'p1'
                    color= 'white'
                elif data[i, j] == 2.:
                    data_grid = 'p2'
                    color = 'white'
                elif data[i, j] == 1. and title == 'semantic similar group':
                    data_grid = 'p3'
                    color = 'white'
                else:
                    data_grid = data[i, j]
                    color = textcolors[int(im.norm(data[i, j]) > threshold)]
            kw.update(color=color)
            if isinstance(data_grid, str):
                text = im.axes.text(j, i, data_grid, **kw)
            else:
                text = im.axes.text(j, i, valfmt(data_grid, None), **kw)
            texts.append(text)

    return texts

def visualize_model(models, dataloader, device, num_images=6, set='test', cls=None, save_img=False):
    print('Visualizing model...')
    assert set in ('train', 'test')

    for model in models:
        model.eval()
    images_so_far = 0

    #names = ['source', 'GMN', 'PCA-GM', 'IPCA-GM']
    #names = ['source', 'GMN', 'PCA-GM', 'NGM', 'NGM-v2', 'NHGM-v2']
    names = ['source', 'GMN', 'PCA-GM', 'CIE', 'GANN-GM', 'GANN-MGM']
    num_cols = num_images #// 2 #+ 1

    old_cls = dataloader[set].dataset.cls
    if cls is not None:
        dataloader[set].dataset.cls = cls

    visualize_path = Path(cfg.OUTPUT_PATH) / 'visual'
    if save_img:
        if not visualize_path.exists():
            visualize_path.mkdir(parents=True)

    for cls in range(3):
        fig = plt.figure(figsize=(60, 30), dpi=120)
        dataloader[set].dataset.cls = cls

        images_so_far = 0

        for i, inputs in enumerate(dataloader[set]):
            if models[0].module.device != torch.device('cpu'):
                inputs = data_to_cuda(inputs)
            assert 'images' in inputs
            two_gm_inputs = {}
            for key, val in inputs.items():
                if key == 'gt_perm_mat':
                    #two_gm_inputs[key] = torch.bmm(val[0], val[1].transpose(1, 2))
                    two_gm_inputs[key] = val
                elif key == 'KGHs':
                    #two_gm_inputs[key] = val['0,1']
                    two_gm_inputs[key] = val
                elif key == 'num_graphs':
                    two_gm_inputs[key] = 2
                else:
                    if isinstance(val, list) and len(val) > 2:
                        two_gm_inputs[key] = val[0], val[1]
                    else:
                        two_gm_inputs[key] = val

            data1, data2 = two_gm_inputs['images']
            P1_gt, P2_gt = two_gm_inputs['Ps']
            n1_gt, n2_gt = two_gm_inputs['ns']
            perm_mat = two_gm_inputs['gt_perm_mat']

            pred_perms = []
            for model_id, model in enumerate(models):
                cfg.PROBLEM.TYPE = '2GM'
                outputs = model(inputs)
                pred_perms.append(outputs['perm_mat'])
                '''
                if model_id != len(models) - 1:
                    cfg.PROBLEM.TYPE = '2GM'
                    outputs = model(two_gm_inputs)
                    pred_perms.append(outputs['perm_mat'])
                else:
                    cfg.PROBLEM.TYPE = 'MGM'
                    outputs = model(inputs)
                    for (idx1, idx2), pred_perm in zip(outputs['graph_indices'], outputs['perm_mat_list']):
                        if idx1 == 0 and idx2 == 1:
                            pred_perms.append(pred_perm)
                            break
                        elif idx1 == 1 and idx2 == 0:
                            pred_perms.append(pred_perm.transpose(1, 2))
                            break
                '''

            for j in range(inputs['batch_size']):
                if n1_gt[j] <= 4:
                    print('graph too small.')
                    continue

                matched = []
                for idx, pred_perm in enumerate(pred_perms):
                    matched_num = torch.sum(pred_perm[j, :n1_gt[j], :n2_gt[j]] * perm_mat[j, :n1_gt[j], :n2_gt[j]])
                    matched.append(matched_num)

                #if random.choice([0, 1, 2]) >= 1:
                '''
                if not (matched[4] >= matched[3] >= matched[1] > matched[0]):
                        print('performance not good.')
                        continue
                '''

                images_so_far += 1
                print(chr(13) + 'Visualizing {:4}/{:4}'.format(images_so_far, num_images))  # chr(13)=CR

                colorset = np.random.rand(n1_gt[j], 3)
                #ax = plt.subplot(1 + len(s_pred_perms), num_cols, images_so_far + 1)
                #ax.axis('off')
                #plt.title('source')
                #plot_helper(data1[j], P1_gt[j], n1_gt[j], ax, colorset)

                for idx, pred_perm in enumerate(pred_perms):
                    ax = plt.subplot(len(pred_perms), num_cols, idx * num_cols + images_so_far)
                    #if images_so_far > num_cols:
                    #    ax = plt.subplot(len(pred_perms) * 2, num_cols, (idx + len(pred_perms)) * num_cols + images_so_far - num_cols)
                    #else:
                    #    ax = plt.subplot(len(pred_perms) * 2, num_cols, idx * num_cols + images_so_far)
                    ax.axis('off')
                    #plt.title('predict')
                    #plot_helper(data2[j], P2_gt[j], n1_gt[j], ax, colorset, 'tgt', s_pred_perm[j], perm_mat[j])
                    plot_2graph_helper(data1[j], data2[j], P1_gt[j], P2_gt[j], n1_gt[j], n2_gt[j], ax, colorset, pred_perm[j], perm_mat[j], names[idx+1], cls=dataloader[set].dataset.cls)

                #ax = plt.subplot(2 + len(s_pred_perms), num_images + 1, (len(s_pred_perms) + 1) * num_images + images_so_far)
                #ax.axis('off')
                #plt.title('groundtruth')
                #plot_helper(data2[j], P2_gt[j], n1_gt[j], ax, colorset, 'tgt', perm_mat[j])

                if not save_img:
                    plt.show()
                    print("Press Enter to continue...", end='', flush=True)  # prevent new line
                    input()

                if images_so_far >= num_images:
                    # fig.savefig(str(visualize_path / '{}_{:0>4}.jpg'.format(dataloader[set].dataset.cls, images_so_far)), bbox_inches='tight')
                    fig.savefig(str(visualize_path / '{}_{:0>4}.jpg'.format(dataloader[set].dataset.cls, images_so_far)))
                    break

                #dataloader[set].dataset.cls += 1
            if images_so_far >= num_images:
                break

    dataloader[set].dataset.cls = old_cls


def plot_helper(img, P, n, ax, colorset, mode='src', pmat=None, gt_pmat=None):
    assert mode in ('src', 'tgt')
    if mode == 'tgt':
        assert pmat is not None
    img = tensor2np(img.cpu())
    plt.imshow(img)

    P = P.cpu().numpy()
    if mode == 'src':
        for i in range(n):
            mark = plt.Circle(P[i], 7, edgecolor='w', facecolor=colorset[i])
            ax.add_artist(mark)
    else:
        pmat = pmat.cpu().numpy()
        gt_pmat = gt_pmat.cpu().numpy()
        idx = np.argmax(pmat, axis=-1)
        idx_gt = np.argmax(gt_pmat, axis=-1)
        matched = 0
        for i in range(n):
            mark = plt.Circle(P[idx[i]], 7, edgecolor='w' if idx[i] == idx_gt[i] else 'r', facecolor=colorset[i])
            ax.add_artist(mark)
            if idx[i] == idx_gt[i]:
                matched += 1
        plt.title('{:d}/{:d}'.format(matched, n), y=-0.2, fontsize=25)


def plot_triangle(ax, A, P, n , perturb=False):
    '''
    Inputs:
        A1, A2: the adjancy matrix of graph1 and graph2
        P1, P2: the position of vertices in images of graph1 and 2
        n1, n2: the number of vertices in graph1 and graph2
    output:
        Draw the graph in the images
    '''
    linewidth = 1
    for i in range(n):
        for j in range(i, n, 1):
            if A[i,j] == 1 : 
                l = matplotlib.lines.Line2D([P[i][0], P[j][0]], [P[i][1], P[j][1]],
                                             linewidth=linewidth,  color='#069AF3')
                ax.add_line(l)
                    

def plot_image_helper(imgsrc, imgtgt):
    imgcat = torch.cat((imgsrc, imgtgt), dim=2)
    imgcat = tensor2np(imgcat.cpu())
    plt.imshow(imgcat)

def plot_node_helper(ax, Psrc, Ptgt, nsrc, ntgt, pmat, perturb=False):
    radius = 5
    eps = 35 

    for i in range(nsrc):
        mark = plt.Circle(Psrc[i], radius, edgecolor='#069AF3', facecolor="None")
        ax.add_artist(mark)
        if perturb:
            rou = 2. * math.pi * (random.random() - 0.5)
            ax.arrow(*Psrc[i], math.cos(rou) * eps, math.sin(rou) * eps,
                    overhang=0.3, width=0.05, 
                    head_width=9, head_length=13, color='#00ED92')
            #plt.show()

    for j in range(ntgt):
        mark = plt.Circle(Ptgt[j], radius, edgecolor='#069AF3', facecolor="None")
        ax.add_artist(mark)
        if perturb:
            rou = 2. * math.pi * (random.random() - 0.5)
            ax.arrow(*Ptgt[j],  math.cos(rou) * eps, math.sin(rou) * eps,
                    overhang=0.3, width=0.05, 
                    head_width=9, head_length=13, color='#06F39A')
            #plt.show()

def plot_2graph_helper(imgsrc, imgtgt, Psrc, Ptgt, nsrc, ntgt, ax, colorset, pmat, gt_pmat, dist_mat, method="", cls=0, changed_pos=None, Asrc=None, Atgt=None, img=True, match=True, perturb=False):
    if img: 
        plot_image_helper(imgsrc, imgtgt)
    else:
        plot_image_helper(torch.ones_like(imgsrc) * 255, torch.ones_like(imgtgt) * 255)

    # import pdb; pdb.set_trace()
    matched = 0
    radius = 10
    linewidth = 2
    #fontsize = 40
    #font_dict['size'] = fontsize
    Psrc = Psrc.cpu().detach().numpy()
    Ptgt = Ptgt.cpu().detach().numpy()
    Ptgt[:, 0] += imgsrc.shape[2]
    pmat = pmat.cpu().detach().numpy()
    gt_pmat = gt_pmat.cpu().detach().numpy()
    dist_mat = dist_mat.cpu().detach().numpy()

    if perturb:
        plot_node_helper(ax, Psrc, Ptgt, nsrc, ntgt, pmat, perturb=perturb)

    if Asrc is not None:
        plot_triangle(ax, Asrc, Psrc, nsrc, pmat)
    if Atgt is not None:
        plot_triangle(ax, Atgt, Ptgt, ntgt, pmat)
    
    if changed_pos is not None:
        mark = plt.Circle(changed_pos, radius, edgecolor='y', facecolor='None', linestyle='-.')
        ax.add_artist(mark)

    if match:
        for i in range(nsrc):
            for j in range(ntgt):
                if pmat[i, j] == 1:
                    # src
                    mark = plt.Circle(Psrc[i], radius, edgecolor='g' if gt_pmat[i, j] == 1 else 'r', facecolor="None", linewidth=linewidth)
                    ax.add_artist(mark)
                    #tgt
                    mark = plt.Circle(Ptgt[j], radius, edgecolor='g' if gt_pmat[i, j] == 1  else 'r', facecolor="None", linewidth=linewidth)
                    ax.add_artist(mark)
                    # l = matplotlib.lines.Line2D([Psrc[i][0], Ptgt[j][0]], [Psrc[i][1], Ptgt[j][1]], linewidth=linewidth, linestyle='dashed', color='g' if gt_pmat[i, j] == 1 else 'r')
                    l = matplotlib.lines.Line2D([Psrc[i][0], Ptgt[j][0]], [Psrc[i][1], Ptgt[j][1]], linewidth=linewidth, color='g' if gt_pmat[i, j] == 1 else 'r')
                    ax.add_line(l)
                    # randomly annotate predicted matching probability on any of two graphs 
                    # if random.random() < 0.5:
                    #ax.text(Psrc[i][0], Psrc[i][1], dist_mat[i][j].round(2), fontdict=font_dict, bbox=dict(facecolor='g' if gt_pmat[i, j] == 1 else 'r', alpha=0.5))
                    # else:
                    #     ax.text(Ptgt[j][0], Ptgt[j][1], dist_mat[i][j].round(2), style='italic', fontsize=fontsize, color='w', bbox=dict(facecolor='red', alpha=0.5))
                    # import pdb; pdb.set_trace()
                    # ax.add_artist(text)
                    if gt_pmat[i, j] == 1:
                        matched += 1
    #plt.title('{} {}: {:d}/{:d}'.format(method, cfg.IMC_PT_SparseGM.CLASSES['test'][dataloader['test'].dataset.cls], matched, round(gt_pmat.sum())), y=-0.3, fontsize=20)
    plt.title('{} {}: {:d}/{:d}'.format(method, eval('cfg.'+cfg.DATASET_FULL_NAME).CLASSES[cls], matched, int(round(gt_pmat.sum()))), fontsize=38, y=-0.3)
    # plt.title('{} {}: {:d}/{:d}'.format(method, eval('cfg.'+cfg.DATASET_FULL_NAME).CLASSES[cls], matched, int(round(gt_pmat.sum()))), y=-0.3, fontsize=30)

def tensor2np(inp):
    """Tensor to numpy array for plotting"""
    inp = inp.detach().numpy().transpose((1, 2, 0))
    mean = np.array(cfg.NORM_MEANS)
    std = np.array(cfg.NORM_STD)
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp


if __name__ == '__main__':
    from src.utils.parse_args import parse_args
    args = parse_args('Deep learning of graph matching visualization code.')
    
    import importlib
    from src.utils.config import cfg_from_file

    dataset_len = {'train': cfg.TRAIN.EPOCH_ITERS * cfg.BATCH_SIZE, 'test': cfg.EVAL.SAMPLES}
    image_dataset = {
        x: GMDataset(cfg.DATASET_FULL_NAME,
                     sets=x,
                     problem=cfg.PROBLEM.TYPE,
                     length=dataset_len[x],
                     cls=cfg.TRAIN.CLASS if x == 'train' else cfg.EVAL.CLASS,
                     obj_resize=cfg.PROBLEM.RESCALE)
        for x in ('train', 'test')}
    cfg.DATALOADER_NUM = 0
    dataloader = {x: get_dataloader(image_dataset[x], fix_seed=(x == 'test'))
        for x in ('train', 'test')}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_paths = ['/home/wangrunzhong/dl-of-gm/pretrained_weights/pretrained_params_vgg16_gmn_imcpt.pt',
                   '/home/wangrunzhong/dl-of-gm/pretrained_weights/pretrained_params_vgg16_pca_imcpt.pt',
                   '/home/wangrunzhong/dl-of-gm/pretrained_weights/pretrained_params_vgg16_cie_imcpt.pt',
                   '/home/wangrunzhong/dl-of-gm/pretrained_weights/pretrained_params_vgg16_gann-gm_imcpt.pt',
                   '/home/wangrunzhong/dl-of-gm/pretrained_weights/pretrained_params_vgg16_gann-mgm_imcpt.pt',
                   ]

    cfg_files = ['experiments/vgg16_gmn_imcpt.yaml',
                 'experiments/vgg16_pca_imcpt.yaml',
                 'experiments/vgg16_cie_imcpt.yaml',
                 'experiments/vgg16_gann-gm_imcpt.yaml',
                 'experiments/vgg16_gann-mgm_imcpt.yaml',
                 ]
    models = []

    model_paths = ['/mnt/nas/home/wangrunzhong/dl-of-gm/pretrained_weights/pretrained_params_vgg16_ngm_voc.pt']
    cfg_files = ['experiments/vgg16_ngm_voc.yaml']
    for i, (model_path, cfg_file) in enumerate(zip(model_paths, cfg_files)):
        cfg_from_file(cfg_file)

        mod = importlib.import_module(cfg.MODULE)
        Net = mod.Net

        model = Net()
        model = model.to(device)
        model = DataParallel(model, device_ids=cfg.GPUS)

        print('Loading model parameters from {}'.format(model_path))
        load_model(model, model_path)
        models.append(model)

    visualize_model(models, dataloader, device,
                    num_images=1,
                    cls=None,
                    save_img=True)
    '''
    visualize_model(models, dataloader, device,
                    num_images=cfg.VISUAL.NUM_IMGS,
                    cls=cfg.VISUAL.CLASS if cfg.VISUAL.CLASS != 'none' else None,
                    save_img=cfg.VISUAL.SAVE)
    '''
