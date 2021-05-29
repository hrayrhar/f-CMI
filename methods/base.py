from nnlib.nnlib import visualizations as vis
from nnlib.nnlib.method_utils import Method


class BaseClassifier(Method):
    """ Abstract class for classifiers.
    """
    def __init__(self, **kwargs):
        super(BaseClassifier, self).__init__(**kwargs)

    def visualize(self, train_loader, val_loader, **kwargs):
        visualizations = {}

        # visualize pred
        fig, _ = vis.plot_predictions(self, train_loader, key='pred')
        visualizations['predictions/pred-train'] = fig
        if val_loader is not None:
            fig, _ = vis.plot_predictions(self, val_loader, key='pred')
            visualizations['predictions/pred-val'] = fig

        return visualizations
