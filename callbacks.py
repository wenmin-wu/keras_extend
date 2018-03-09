import tensorflow as tf
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
    """
    add summaries to TensorBoard on batch end.
    """
    def __init__(self, log_dir='./logs',
                 histogram_freq=0,
                 batch_size=32,
                 write_graph=True,
                 write_grads=False,
                 write_images=False,
                 embeddings_freq=0,
                 embeddings_layer_names=None,
                 embeddings_metadata=None):
        super(BatchTensorBoard, self).__init__(log_dir,
                                               histogram_freq,
                                               batch_size,
                                               write_graph,
                                               write_grads,
                                               write_images,
                                               embeddings_freq,
                                               embeddings_layer_names,
                                               embeddings_metadata)
        self.seen = 0

    #TODO: find a better to deal with this
    def on_epoch_end(self, epoch, logs=None):
        """
        to avoid the summary being overrite
        """
        pass

    def on_batch_end(self, batch, logs={}):
        for name, value in logs.items():
            if name == 'batch' or name == 'size':
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, self.seen)
        self.writer.flush()
        self.seen += self.batch_size
