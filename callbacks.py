from keras.callbacks import Callback
from keras.callbacks import TensorBoard

class NBatchLogger(Callback):
    """
    A logger which log average performaces per `display` batches
    """
    def __init__(self, display):
        self.curr_step = 0
        self.display = display
        self.metric_cache = {}

    def on_batch_end(self, batch, logs={}):
        self.step += 1
        metrics = self.params['metrics']
        for k in metrics:
            if k in logs:
                self.metric_cache.setdefault(k, 0)
                self.metric_cache[k] += logs[k]

        if self.step % self.display == 0:
            metrics_log = ''
            for (k, v) in self.metric_cache.items():
                val = v / self.display
                if abs(val) > 1e-3:
                    metrics_log += ' - %s: %.4f' % (k, val)
                else:
                    metrics_log += ' - %s: %.4f' % (k, val)
            print('step: {}/{} ...{}'.format(self.step,
                                             self.params['steps'],
                                             metrics_log)
                  )
            self.metric_cache.clear()

class BatchTensorBoard(TensorBoard):
    pass
