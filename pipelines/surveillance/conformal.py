import numpy as np

class ConformalPredictor:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
    
    def calibrate(self, y_true, y_pred, weights=None):
        pass

    def predict(self, y_pred, weights=None):
        # Placeholder intervals: +/- 10%
        return y_pred - 0.1, y_pred + 0.1
