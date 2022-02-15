from basars_addons.metrics import ThresholdBinaryIoU


class MaskedThresholdBinaryIoU(ThresholdBinaryIoU):
    def __init__(self, num_classes, mask, inverse=False, threshold=0.5, name=None, dtype=None):
        super(MaskedThresholdBinaryIoU, self).__init__(num_classes, threshold, name, dtype)

        self.mask = mask
        self.inverse = inverse

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = y_true[:, :, :, self.mask]
        y_pred = y_pred[:, :, :, self.mask]

        if self.inverse:
            y_true = 1. - y_true
            y_pred = 1. - y_pred

        return super(MaskedThresholdBinaryIoU, self).update_state(y_true, y_pred, sample_weight)
