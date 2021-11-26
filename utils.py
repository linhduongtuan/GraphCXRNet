import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch

from torch_geometric.loader import DataLoader

################################################################################
# METRICS



def plot_cm(cm, 
            display_labels= ['BIRAD_0', 'BIRAD_1', 'BIRAD_2', 'BIRAD_3','BIRAD_4A', 'BIRAD_4B','BIRAD_4C', 'BIRAD_5']):

    """Plot confusion matrix with heatmap.
    Args:
        cm : array
            Confusion matrix
        display_labels : list, optional
            Labels of classes in confusion matrix, by default ["Mutag", "Non Mutag"]
    """
    # Set fontsize for plots
    font = {"size": 20}
    matplotlib.rc("font", **font)

    # Plot confusion matrix
    f, axes = plt.subplots(1, 1, figsize=(7, 7), sharey="row")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(ax=axes, xticks_rotation=45, cmap="Blues", values_format='d')
    disp.im_.colorbar.remove()
    disp.ax_.set_xlabel("Predicted label", fontsize=20)
    disp.ax_.set_ylabel("True label", fontsize=20)
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    plt.show()




