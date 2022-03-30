#!/usr/bin/env python3
import numpy as np
import diamond_square
import torch
import timeit
import matplotlib.pyplot as plt
import os
import fargv
import seaborn as sns
from matplotlib import pyplot as plt
from collections import defaultdict

# The array must be square with edge length 2**n + 1

def ds_cpp(n):
    os.system(f"./third_party/DiamondSquare/bin/DiamondSquare {n} no_io")

def ds_tormentor_cpu(n):
    res = diamond_square.diamond_square((2**n+1, 2**n+1), device="cpu")
    return res.numpy()

def ds_tormentor_gpu(n):
    try:
        res = diamond_square.diamond_square((2**n+1, 2**n+1), device="cuda")
        return res
    except RuntimeError:
        return None

def ds_np(n):
    """
    
    Base on an implementation from:
    https://scipython.com/blog/cloud-images-using-the-diamond-square-algorithm/
    Returns:

    """
    N = 2**n + 1
    # f scales the random numbers at each stage of the algorithm
    f = 1.0

    # Initialise the array with random numbers at its corners
    arr = np.zeros((N, N))
    arr[0::N-1,0::N-1] = np.random.uniform(-1, 1, (2,2))
    side = N-1

    nsquares = 1
    while side > 1:
        sideo2 = side // 2

        # Diamond step
        for ix in range(nsquares):
            for iy in range(nsquares):
                x0, x1, y0, y1 = ix*side, (ix+1)*side, iy*side, (iy+1)*side
                xc, yc = x0 + sideo2, y0 + sideo2
                # Set this pixel to the mean of its "diamond" neighbours plus
                # a random offset.
                arr[yc,xc] = (arr[y0,x0] + arr[y0,x1] + arr[y1,x0] + arr[y1,x1])/4
                arr[yc,xc] += f * np.random.uniform(-1,1)

        # Square step: NB don't do this step until the pixels from the preceding
        # diamond step have been set.
        for iy in range(2*nsquares+1):
            yc = sideo2 * iy
            for ix in range(nsquares+1):
                xc = side * ix + sideo2 * (1 - iy % 2)
                if not (0 <= xc < N and 0 <= yc < N):
                    continue
                tot, ntot = 0., 0
                # Set this pixel to the mean of its "square" neighbours plus
                # a random offset. At the edges, it has only three neighbours
                for (dx, dy) in ((-1,0), (1,0), (0,-1), (0,1)):
                    xs, ys = xc + dx*sideo2, yc + dy*sideo2
                    if not (0 <= xs < N and 0 <= ys < N):
                        continue
                    else:
                        tot += arr[ys, xs]
                        ntot += 1
                arr[yc, xc] += tot / ntot + f * np.random.uniform(-1,1)
        side = sideo2
        nsquares *= 2
        f /= 2
    return arr

def show(arr):
    plt.imshow(arr, cmap=plt.cm.Blues)
    plt.axis('off')
    plt.show()



def time_experiments(n, replicates):
    MP=1
    MP = ((2 ** n + 1) ** 2) / 1000.0
    cpp = sorted([t/MP for t in timeit.Timer(f"a=ds_cpp({n})", globals = globals()).repeat(replicates+2,number=1)])[1:-1]
    numpy = sorted([t/MP for t in timeit.Timer(f"a=ds_np({n})", globals = globals()).repeat(replicates+2,number=1)])[1:-1]
    tormentor_cpu = sorted([t/MP for t in timeit.Timer(f"a=ds_tormentor_cpu({n})", globals = globals()).repeat(replicates+2,number=1)])[1:-1]
    if torch.cuda.is_available():
        tormentor_gpu = sorted([t/MP for t in timeit.Timer(f"a=ds_tormentor_gpu({n})", globals=globals()).repeat(replicates+2, number=1)])[1:-1]
    else:
        tormentor_gpu = [0.0]
    MP = ((2 ** n + 1) ** 2) / 1000000.0
    del n, replicates
    return locals()




def plot(aggregated_results, title, tick="MP", y_label="Duration (msec. / MPixels)", x_label="Image size (MPixels)", captions={"cpp":"C++",
                                                                                            "numpy": "Numpy",
                                                                                            "tormentor_cpu":"Proposed (CPU)",
                                                                                            "tormentor_gpu":"Proposed (GPU)"}):
    modalities = set(aggregated_results.keys()) - set([tick])
    ticks = aggregated_results[tick]
    for name in modalities:
        values = np.array(aggregated_results[name])
        print(name, values.shape)
        caption = captions[name]
        #sns.lineplot(x=ticks, y=values.mean(axis=1), label=caption)
        ci = "sd"
        #plt.errorbar(x=ticks, y=values.mean(axis=1), xerr=values.std(axis=1), linestyle='None', marker='^')
        #plt.errorbar(ticks, values.mean(axis=1), yerr=values.std(axis=1), marker='^')
        #plt.barplot(ticks, values.mean(axis=1), yerr=values.std(axis=1), marker='^')
        #sns.relplot(x=name, y=tick, kind="line", ci="sd", data=aggregated_results)
        m = values.mean(axis=1)
        v = values.std(axis=1)
        plt.plot(ticks, values.mean(axis=1), label=caption)
        #plt.errorbar(y=ticks, x=values.mean(axis=1), yerr=values.std(axis=1))
        plt.fill_between(ticks, m-v, m+v, alpha=0.5)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.title(title)
    plt.yscale("log")
    plt.xscale("log")


p={
    "n":"6,7,8,9,10,11,12,13",
    "replicates": 1,
    "plot": True,
    "table": True,
    "plot_fname":"./tmp/ds_time.pdf",
    "table_fname":"./tmp/table.tex",
}

if __name__ == "__main__":
    sns.set_theme()
    p, _ = fargv.fargv(p)
    all_n = [int(n) for n in p.n.split(",")]
    aggregated_results = defaultdict(lambda:[])
    for n in all_n:
        measurements = time_experiments(n, p.replicates)
        for k, v in measurements.items():
            aggregated_results[k].append(v)
    aggregated_results = {k: np.array(v) for k, v in aggregated_results.items()}
    if p.plot:
        plot(aggregated_results,"Diamond Square Running Times", tick="MP")
        if p.plot_fname == "":
            plt.show()
        else:
            plt.savefig(p.plot_fname)