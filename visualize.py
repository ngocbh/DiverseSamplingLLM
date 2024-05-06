import matplotlib.pyplot as plt
from metrics import metric_map


def plot(rescols, labels, x_metric, y_metric, x_scale='linear', y_scale='linear', save_path=None, show=False):
    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots()
    for rescol, label in zip(rescols, labels):
        df = rescol.get_df()
        x = df[x_metric].to_numpy()
        y = df[y_metric].to_numpy()
        ax.plot(x, y, label=label)
    ax.set_xlabel(metric_map[x_metric])
    ax.set_ylabel(metric_map[y_metric])
    ax.set_yscale(y_scale)
    ax.set_xscale(x_scale)
    ax.legend()
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=400)
    
    if show:
        plt.show()
    
