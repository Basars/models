from basars_addons.metrics import ThresholdPrecision, ThresholdRecall


class MaskedThresholdPrecision(ThresholdPrecision):

    def __init__(self,
                 mask, inverse=False,
                 threshold=0.5, thresholds=None, top_k=None, class_id=None, from_logits=False,
                 name=None, dtype=None):
        super(MaskedThresholdPrecision, self).__init__(threshold, thresholds, top_k, class_id, from_logits, name, dtype)

        self.mask = mask
        self.inverse = inverse

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = y_true[:, :, :, self.mask]
        y_pred = y_pred[:, :, :, self.mask]

        if self.inverse:
            y_true = 1. - y_true
            y_pred = 1. - y_pred

        return super(MaskedThresholdPrecision, self).update_state(y_true, y_pred, sample_weight)


class MaskedThresholdRecall(ThresholdRecall):

    def __init__(self,
                 mask, inverse=False,
                 threshold=0.5, thresholds=None, top_k=None, class_id=None, from_logits=False,
                 name=None, dtype=None):
        super(MaskedThresholdRecall, self).__init__(threshold, thresholds, top_k, class_id, from_logits, name, dtype)

        self.mask = mask
        self.inverse = inverse

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = y_true[:, :, :, self.mask]
        y_pred = y_pred[:, :, :, self.mask]

        if self.inverse:
            y_true = 1. - y_true
            y_pred = 1. - y_pred

        return super(MaskedThresholdRecall, self).update_state(y_true, y_pred, sample_weight)
