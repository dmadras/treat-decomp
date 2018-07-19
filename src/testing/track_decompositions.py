import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


RESFNAME = 'valid_results.csv'

def load_data(resf):
    f = open(resf, 'r')
    headers = f.readline().strip().split(',')
    lines = [line.strip().split(',') for line in f.readlines()]
    info = np.array([[float(x) for x in l] for l in lines])
    return info, headers

def get_indices(ind_names, headers):
    return [headers.index(ind_name) for ind_name in ind_names]

def plot_group(x, inds, ind_names, settings, figname):
    plt.clf()
    for i, i_name in zip(inds, ind_names):
        plot_data = x[:,i]
        plt.plot(plot_data, label=i_name, **settings['plot'])
    plt.legend()
    # plt.ylim([np.percentile(x[:,inds], 1), np.percentile(x[:,inds], 96)])
    plotdat = x[4:,inds]
    plt.ylim([np.min(plotdat) - np.std(plotdat), np.max(plotdat) + np.std(plotdat)])
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.savefig(figname)

def plot_pairs(x, inds, ind_names, settings, figname):
    plt.clf()
    for i, i_name in zip(inds, ind_names):
        plot_data_0 = x[:,i[0]]
        plot_data_1 = x[:,i[1]]
        plot_data = plot_data_1 - plot_data_0
        plt.plot(plot_data, label=i_name, **settings['plot'])
    plt.legend()
    plotdat = np.concatenate([x[9:,i[1]] - x[9:,i[0]] for i in inds])
    plt.ylim([np.min(plotdat) - np.std(plotdat), np.max(plotdat) + np.std(plotdat)])
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value - Diff (1 - 0)')
    plt.savefig(figname)

def main(dname, figdir, track_groups, track_pairs, settings):
    resf = os.path.join(dname, RESFNAME)
    info, headers = load_data(resf)
    for group in track_groups:
        plot_indices = get_indices(track_groups[group], headers)
        figname = os.path.join(figdir, '{}_by_epoch.png'.format(group))
        plot_group(info, plot_indices, track_groups[group], settings, figname)
    for pair_name in track_pairs:
        plot_indices = [get_indices(track_pairs[pair_name][met], headers) for met in track_pairs[pair_name]] 
        metric_names = ['{}-diff'.format(met) for met in track_pairs[pair_name]]
        figname = os.path.join(figdir, '{}-diff_by_epoch.png'.format(pair_name))
        plot_pairs(info, plot_indices, metric_names, settings, figname)


