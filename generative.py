import numpy as np
import igraph as ig
import dionysus as d
import multiprocessing
# from dask.distributed import Client
from zigzag import sliding_windows, zigzag_network
from wrcf import wrcf_diagram
from sliced_wasserstein import diagram_array, SW_approx
import dill


def random_edge_presences(T, f):
    """Generate random times sampled over a periodic distribution.
    :param T: time range
    :param f: frequency
    :return: an array of times.
    """
    density = np.sin(f * np.arange(T)) + 1
    density /= np.sum(density)
    samplesize = np.random.randint(T//2)
    times = np.random.choice(np.arange(T), size=samplesize, replace=False, p=density)
    times = np.sort(times)
    return times


def remove_inf(dgm):
    """Remove infinite points in a persistence diagram.
    :param dgm: Diagram
    :return: the same diagram without the infinite points.
    """
    res = d.Diagram()
    for p in dgm:
        if p.death != np.inf:
            res.append(p)
    return res


## Global parameters
NODES = 40
EDGE_PROB = 0.9
TIME_RANGE = 200
FREQ = 15/TIME_RANGE
N_WINDOWS = 20
## Computations
ZIGZAG_PERS = True
WRCF_PERS = True
SW_KERNEL = True
BOTTLENECK_DIST = True


if __name__=="__main__":

    print("Generating random temporal network...", end="", flush=True)

    basegraph = ig.Graph.Erdos_Renyi(NODES, EDGE_PROB)
    g = ig.Graph()
    g.add_vertices(len(basegraph.vs))

    for e in basegraph.es:
        times = random_edge_presences(TIME_RANGE, FREQ)
        for t in times:
            g.add_edge(e.source, e.target, time=t)
    print("done.")

    print("Temporal partitioning...", end="", flush=True)
    wins = sliding_windows(g, 1/N_WINDOWS)
    print("done.")

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    if ZIGZAG_PERS:
        print("Zigzag persistence...", end="", flush=True)
        zz_dgms = pool.map(zigzag_network, wins)
        dill.dump(zz_dgms, open("generative/zz_dgms.dill", "wb"))
        print("done, saved.")

    if WRCF_PERS:
        print("WRCF...", end="", flush=True)
        ## Collapse each subnetwork into a static graph: the weight is the
        ## number of appearances of each edge
        for w in wins:
            w.es["time"] = np.repeat(1, len(w.es["time"]))
            w.simplify(combine_edges="sum")
            w.es["weight"] = w.es["time"]
            del w.es["time"]
        wrcf_dgms = pool.map(wrcf_diagram, wins)
        dill.dump(wrcf_dgms, open("generative/wrcf_dgms.dill", "wb"))
        print("done.")

    pool.terminate()

    if ZIGZAG_PERS and SW_KERNEL:
        print("Sliced Wasserstein Kernel (zigzag)...", end="", flush=True)
        zz_dgms1 = [dgm[1] for dgm in zz_dgms if len(dgm) > 1]
        zz_gram1 = np.array([[SW_approx(zz_dgms1[i], zz_dgms1[j], 10)
                              for i in range(len(zz_dgms1))] for j in range(len(zz_dgms1))])
        dill.dump(zz_gram1, open("generative/zz_gram1.dill", "wb"))
        print("done, saved.")

    if WRCF_PERS and SW_KERNEL:
        print("Sliced Wasserstein Kernel (WRCF)...", end="", flush=True)
        wrcf_dgms1 = [dgm[1] for dgm in wrcf_dgms if len(dgm) > 1]
        wrcf_gram1 = np.array([[SW_approx(wrcf_dgms1[i], wrcf_dgms1[j], 10)
                                for i in range(len(wrcf_dgms1))] for j in range(len(wrcf_dgms1))])
        dill.dump(wrcf_gram1, open("generative/wrcf_gram1.dill", "wb"))
        print("done, saved.")

    if ZIGZAG_PERS and BOTTLENECK_DIST:
        print("Bottleneck distance (zigzag)...", end="", flush=True)
        zz_dgms1 = list(map(remove_inf, zz_dgms1))
        zz_distmat = np.array([[d.bottleneck_distance(zz_dgms1[i], zz_dgms1[j])
                                for i in range(len(zz_dgms1))] for j in range(len(zz_dgms1))])
        dill.dump(zz_distmat, open("generative/zz_distmat.dill", "wb"))
        print("done, saved.")

    if WRCF_PERS and BOTTLENECK_DIST:
        print("Bottleneck distance (WRCF)...", end="", flush=True)
        wrcf_dgms1 = list(map(remove_inf, wrcf_dgms1))
        wrcf_distmat = np.array([[d.bottleneck_distance(wrcf_dgms1[i], wrcf_dgms1[j])
                                  for i in range(len(wrcf_dgms1))] for j in range(len(wrcf_dgms1))])
        dill.dump(wrcf_distmat, open("generative/wrcf_distmat.dill", "wb"))
        print("done, saved.")
