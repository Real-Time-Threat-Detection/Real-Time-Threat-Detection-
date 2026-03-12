import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable
import seaborn as sn
import pandas as pd
from itertools import cycle
from sklearn.metrics import roc_curve, confusion_matrix, roc_auc_score


def Plot_Results():
    # New color palette and new markers
    color_palette = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']  # Updated colors
    bar_palette = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']  # Same colors for bar plot
    markers = ['*', 'H', 'v', 'X', 'P']  # New markers for each line
    for a in range(2):
        # Load evaluation data
        Eval = np.load('Eval_Batch.npy', allow_pickle=True)[a]

        # Metrics list
        Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score',
                 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR+', 'LR-', 'MCC']

        # Batch sizes and indices for the terms to plot
        learn = [1, 2, 3, 4, 5]
        Graph_Term = [0, 3, 9, 11, 14, 15, 18]

        for j in range(len(Graph_Term)):
            # Initialize graph array
            Graph = np.zeros((Eval.shape[0], Eval.shape[1]))

            # Populate Graph array with evaluation data
            for k in range(Eval.shape[0]):
                for l in range(Eval.shape[1]):
                    if Graph_Term[j] == 20:
                        Graph[k, l] = Eval[k, l, Graph_Term[j] + 4]
                    else:
                        Graph[k, l] = Eval[k, l, Graph_Term[j] + 4] * 100

            # Line Plot
            plt.figure(figsize=(10, 6))
            for idx, (color, marker) in enumerate(zip(color_palette, markers)):
                plt.plot(learn, Graph[:, idx], color=color, linewidth=4, marker=marker,
                         markerfacecolor='white', markersize=8,
                         label=["FFO-ADeepCRF", "FA-ADeepCRF", "NGO-ADeepCRF",
                                "SOA-ADeepCRF", "IRV-SOA-ADeepCRF"][idx])
            plt.xticks(learn, ['4', '8', '16', '32', '48'], fontsize=10)
            plt.xlabel('Batch Size', fontsize=12)
            plt.ylabel(Terms[Graph_Term[j]], fontsize=12)
            plt.grid(alpha=0.3)
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
            plt.tight_layout()
            path1 = f"./Results/Dataset_{a + 1}_{Terms[Graph_Term[j]]}_Line.png"
            plt.savefig(path1)
            plt.show()

            # Bar Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            X = np.arange(5)
            for idx, color in enumerate(bar_palette):
                ax.bar(X + idx * 0.15, Graph[:, idx + 5], color=color, edgecolor='k', width=0.15,
                       label=["ANN", "ADASYN-CNN ", "ANFIS", "DeepCRF", "IRV-SOA-ADeepCRF"][idx])
            ax.set_xticks(X + 0.3)
            ax.set_xticklabels(['4', '8', '16', '32', '48'], rotation=7)
            ax.set_xlabel('Batch Size', fontsize=12)
            ax.set_ylabel(Terms[Graph_Term[j]], fontsize=12)
            ax.grid(alpha=0.3)
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
            plt.tight_layout()
            path2 = f"./Results/Dataset_{a + 1}_{Terms[Graph_Term[j]]}_Bar.png"
            plt.savefig(path2)
            plt.show()


def Plot_table():
    for b in range(2):
        Eval = np.load('Eval_Hidden.npy', allow_pickle=True)[b]
        Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'FOR',
                 'PT',
                 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR+', 'LR-', 'MCC']

        Algorithm = ['TERMS', "FFO-ADeepCRF", "FA-ADeepCRF", "NGO-ADeepCRF",
                     "SOA-ADeepCRF", "IRV-SOA-ADeepCRF"]
        Classifier = ['TERMS', "ANN", "ADASYN-CNN ", "ANFIS", "DeepCRF", "IRV-SOA-ADeepCRF"]
        value = Eval[:, :, 4:]
        value[:, :, :-1] = value[:, :, :-1] * 100

        Graph_Term = [0, 11]
        for a in range(len(Graph_Term)):
            variation = ['100', '200', '300', '400', '500']

            Table = PrettyTable()
            Table.add_column('Hidden Neuron Count /Algorithm', variation[0:])
            for j in range(len(Algorithm) - 1):
                Table.add_column(Algorithm[j + 1], value[:, j, Graph_Term[a]])
            print('---------------------------------------------------- Dataset_', str(b + 1),
                  '- Algorithm Comparison -',
                  Terms[Graph_Term[a]],
                  '--------------------------------------------------')
            print(Table)

            Table = PrettyTable()
            Table.add_column('Hidden Neuron Count /Classifier', variation[0:])
            for j in range(len(Classifier) - 1):
                Table.add_column(Classifier[j + 1], value[:, j + 5, Graph_Term[a]])
            print('---------------------------------------------------Dataset_', str(b + 1), '- Method Comparison -',
                  Terms[Graph_Term[a]],
                  '--------------------------------------------------')
            print(Table)


def Confusion_matrix():
    for a in range(2):
        Actual = np.load('Actual_' + str(a + 1) + '.npy', allow_pickle=True)
        Predict = np.load('Predict_' + str(a + 1) + '.npy', allow_pickle=True)
        ax = plt.subplot()
        cm = confusion_matrix(np.asarray(Actual[4]).argmax(axis=1), np.asarray(Predict[4]).argmax(axis=1))
        sn.heatmap(cm, annot=True, fmt='g',
                   ax=ax).set(title='Confusion Matrix ' + str(a + 1))
        path = "./Results/Confusion_%s.png" % (str(a + 1))
        plt.savefig(path)
        plt.show()


def Plot_ROC():
    lw = 2
    cls = ["ANN", "ADASYN-CNN ", "ANFIS", "DeepCRF", "IRV-SOA-ADeepCRF"]
    colors1 = cycle(["plum", "red", "palegreen", "chocolate", "hotpink", "navy", ])
    colors2 = cycle(["hotpink", "plum", "chocolate", "navy", "red", "palegreen", "violet", "red"])
    for n in range(2):
        for i, color in zip(range(5), colors1):  # For all classifiers
            Predicted = np.load('roc_score.npy', allow_pickle=True)[n][i].astype('float')
            Actual = np.load('roc_act.npy', allow_pickle=True)[n][i].astype('int')
            false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual[:, -1], Predicted[:, -1].ravel())
            roc_auc = roc_auc_score(Actual.ravel(), Predicted.ravel())

            plt.plot(
                false_positive_rate1,
                true_positive_rate1,
                color=color,
                lw=lw,
                label=f'{cls[i]} (AUC = {roc_auc:.3f})',
            )
            #     label="{0}".format(cls[i]),
            # )

        plt.plot([0, 1], [0, 1], "k--", lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        path1 = "./Results/_roc_%s.png" % (str(n + 1))
        plt.savefig(path1)
        plt.show()


def Plot_Fitness():
    for a in range(2):
        conv = np.load('Fitness.npy', allow_pickle=True)[a]
        Statistics = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
        Algorithm = ["FFO-ADeepCRF", "FA-ADeepCRF", "NGO-ADeepCRF",
                     "SOA-ADeepCRF", "IRV-SOA-ADeepCRF"]
        Value = np.zeros((conv.shape[0], 5))
        for j in range(conv.shape[0]):
            Value[j, 0] = np.min(conv[j, :])
            Value[j, 1] = np.max(conv[j, :])
            Value[j, 2] = np.mean(conv[j, :])
            Value[j, 3] = np.median(conv[j, :])
            Value[j, 4] = np.std(conv[j, :])

        Table = PrettyTable()
        Table.add_column("ALGORITHMS", Statistics)
        for j in range(len(Algorithm)):
            Table.add_column(Algorithm[j], Value[j, :])
        print('-------------------------------------------------- Dataset_', str(a + 1),
              '- Statistical Analysis--------------------------------------------------')
        print(Table)

        iteration = np.arange(conv.shape[1])
        plt.plot(iteration, conv[0, :], color='r', linewidth=3, marker='>', markerfacecolor='blue', markersize=8,
                 label="FFO-ADeepCRF")
        plt.plot(iteration, conv[1, :], color='g', linewidth=3, marker='>', markerfacecolor='red', markersize=8,
                 label="FA-ADeepCRF")
        plt.plot(iteration, conv[2, :], color='b', linewidth=3, marker='>', markerfacecolor='green', markersize=8,
                 label="NGO-ADeepCRF")
        plt.plot(iteration, conv[3, :], color='m', linewidth=3, marker='>', markerfacecolor='yellow', markersize=8,
                 label="SOA-ADeepCRF")
        plt.plot(iteration, conv[4, :], color='k', linewidth=3, marker='>', markerfacecolor='cyan', markersize=8,
                 label="IRV-SOA-ADeepCRF")
        plt.xlabel('Iteration')
        plt.ylabel('Cost Function')
        plt.legend(loc=1)
        path1 = "./Results/conv%s.png" % (str(a + 1))
        plt.savefig(path1)
        plt.show()


def New_Plot():
    for a in range(2):
        Eval = np.load('Eval_Hidden.npy', allow_pickle=True)[a]
        Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'FOR',
                 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR+', 'LR-', 'MCC']
        variation = ['100', '200', '300', '400', '500']
        products = ["ANN", "ADASYN-CNN ", "ANFIS", "DeepCRF", "IRV-SOA-ADeepCRF"]
        for b in range(Eval.shape[0]):
            Sensitivity = Eval[b, 5:, 1 + 4]
            Precision = Eval[b, 5:, 3 + 4]
            F1_Score = Eval[b, 5:, 18 + 4]
            colors = ['#f4b183', '#9bbb59', '#c55a11']

            bar_width = 0.20
            bar_gap = 0.02
            x_indices = np.arange(len(products))
            fig, ax = plt.subplots(figsize=(10, 6))

            # Creating bars
            ax.bar(x_indices - (bar_width + bar_gap), Sensitivity, bar_width, color=colors[0], label='Sensitivity',
                   zorder=3)
            ax.bar(x_indices, Precision, bar_width, color=colors[1], label='Precision', zorder=3)
            ax.bar(x_indices + (bar_width + bar_gap), F1_Score, bar_width, color=colors[2], label='MCC', zorder=3)

            ax.set_xlabel('Models', fontsize=14, fontweight='bold', fontfamily='serif')
            ax.set_ylabel('Values', fontsize=14, fontweight='bold', fontfamily='serif')
            ax.set_xticks(x_indices)
            ax.set_xticklabels(products, fontsize=12, fontfamily='serif')
            ax.set_axisbelow(True)
            ax.grid(True, which='major', linestyle='--', linewidth=0.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set(title='Hidden Neuron Count - ' + variation[b] + ' for Dataset - ' + str(a + 1))
            ax.legend()
            plt.tight_layout()
            path = './Results/Dataset - ' + str(a + 1) + '-' + variation[b] + '_bar.png'
            plt.savefig(path)
            plt.show()


def statistical_analysis(v):
    a = np.zeros((5))
    a[0] = np.min(v)
    a[1] = np.max(v)
    a[2] = np.mean(v)
    a[3] = np.median(v)
    a[4] = np.std(v)
    return a


if __name__ == '__main__':
    # Plot_Results()
    # Plot_table()
    # Confusion_matrix()
    Plot_ROC()
    # Plot_Fitness()
    # New_Plot()
