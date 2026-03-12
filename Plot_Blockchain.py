import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable


def statistical_analysis(v):
    a = np.zeros((5))
    a[0] = np.min(v)
    a[1] = np.max(v)
    a[2] = np.mean(v)
    a[3] = np.median(v)
    a[4] = np.std(v)
    return a


no_of_dataset = 2


def plot_results_seg():

    Eval_all = np.load('Block_Chain_Res.npy', allow_pickle=True)
    Terms = ['Transaction Per Seconds', 'Cost of Execution', 'Security Rate', 'Average Latency']

    colors_algo = [ '#4e79a7','#59a14f', '#e15759', '#76b7b2', '#edc948']

    labels_algo = ["BC-ML", "BCAM-DLTM", "BFLS", "IoV-BCFL","BC-APoA"]
    hatch_styles = [ '/','\\', '|', '-', '*']

    for n in range(Eval_all.shape[0]):
        value_all = Eval_all[n, :]
        stats = np.zeros((4, value_all.shape[1], 5))

        for i in range(len(Terms)):
            for j in range(value_all.shape[1]):
                stats[i, j, 0] = np.max(value_all[i,j, :])
                stats[i, j, 1] = np.min(value_all[i,j, :])
                stats[i, j, 2] = np.mean(value_all[i,j, :])
                stats[i, j, 3] = np.median(value_all[i,j, :])
                stats[i, j, 4] = np.std(value_all[i,j, :])



        for k in range(len(Terms)):
            X = np.arange(stats.shape[2])
            bar_width = 0.13

            # ======= ALGORITHMS COMPARISON ========
            plt.figure(figsize=(10, 6))
            for idx, (label, color, hatch) in enumerate(zip(labels_algo, colors_algo, hatch_styles)):
                plt.bar(X + idx * bar_width, stats[k, idx, :], width=bar_width,
                        color=color, edgecolor='black', hatch=hatch, label=label)

            plt.xticks(X + 2 * bar_width, ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD'])
            plt.xlabel('Statistical Measures', fontsize=12)
            plt.ylabel(Terms[k], fontsize=12)
            # plt.title(f'{Terms[Graph_term[k]]} Comparison - {Dataset[n]} (Algorithms)', fontsize=14)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                       ncol=3, fancybox=True, shadow=True)
            plt.tight_layout()
            plt.savefig(f"./Results/Dataset_{n + 1}_Blockchain_{Terms[k]}_Alg.png", dpi=300, bbox_inches='tight')
            plt.show()

plot_results_seg()
