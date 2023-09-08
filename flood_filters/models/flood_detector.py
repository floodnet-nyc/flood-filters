import os
import numpy as np
from ..filter import Filter
from .util import ensure_checkpoint
import onnxruntime as ort


MODELS = {
    '0.0.1': '1Qx9JPSaLQmpg7UNSMqVrvnMqyCImlTPX',
    # '0.1.0': 
    '0.1.0': '1P-qbbDUtbwnGOnqx3TDIjPjQiRSbXax_', # noaug-1-lowsample-weight-20-res2-epoch26
}

DEFAULT_MODEL = '0.1.0'

def _model_path(path):
    if path in MODELS:
        path = ensure_checkpoint(MODELS[path], f'models/{path}.onnx')
    return path or DEFAULT_MODEL

class FloodDetector(Filter):
    def __init__(self, path=DEFAULT_MODEL, buffer_size=None, batch_size=None, mean_window=5, zero_gap=4, name='ml', **kw):
        # print(f'ort avail providers: {ort.get_available_providers()} {ort.get_device()}')
        # self.path = path = _model_path(path)
        # self.model_id = os.path.splitext(os.path.basename(path))[0]
        self.model = Model(path, mean_window=mean_window, zero_gap=zero_gap)

        # # 
        # self.model = ort.InferenceSession(path, providers=ort.get_available_providers())
        # inp = self.model.get_inputs()[0]
        # self.input_name = inp.name
        # self.input_shape = inp.shape
        # self.input_length = buffer_size or inp.shape[1]
        buffer_size = buffer_size or self.model.input.shape[1]
        self.batch_size = batch_size

        # self.mean_window = mean_window
        # self.zero_gap = zero_gap

        assert isinstance(buffer_size, int)
        if batch_size:
            buffer_size = int(buffer_size + batch_size)
        super().__init__(name, buffer_size=buffer_size, min_samples=2, **kw)

    def _filter(self, msg, depth, t):
        # get inputs and pass to model
        x = [d for _, d, _ in self.buffer]
        x = self.preprocess(x)

        # handle batching together for more efficient inference
        msgs, ts = [msg], [t]
        # if self.batch_size and self.batch_size > 1:
        #     self._batch.append((x, msg, t))
        #     if len(self._batch) < self.batch_size:
        #         return
        #     xs, msgs, ts = zip(*self._batch)
        #     x = np.concatenate(xs)

        # run the model
        ys = self.forward(x)

        # post-process outputs
        for y, msg, t in zip(ys, msgs, ts):
            msg = self.postprocess(y, msg)
            yield msg, t

    def _should_hold_point(self, notnull):
        if not notnull:
            return
        self._compute_points()
        return False

    def _should_release_points(self, notnull):
        if not notnull:
            return
        self._compute_points()
        return True

    def _compute_points(self):
        # preprocess the input
        msg, _, t = self.buffer[-1]
        x = [d for _, d, _ in self.buffer]
        ys = self.model(x)
        msg['flood_detected'] = ys[-1]


class Model:
    def __init__(self, path, mean_window=5, zero_gap=4):
        self.path = path = _model_path(path)
        self.model_id = os.path.splitext(os.path.basename(path))[0]

        # 
        self.model = ort.InferenceSession(path, providers=ort.get_available_providers())
        self.input = inp = self.model.get_inputs()[0]
        self.input_name = inp.name
        self.input_shape = inp.shape
        self.input_length = inp.shape[1]

        self.mean_window = mean_window
        self.zero_gap = zero_gap

    def __call__(self, x, agg=True):
        x = self.preprocess(x)
        y = self.forward(x)
        return self.agg(y) if agg else y

    def preprocess(self, x):
        x = np.array(x, dtype=np.float32)

        # drop nan values
        x = x[~np.isnan(x)]

        # look for a gap of zeros and just discard everything before that - to help isolate
        # TODO: could be a convolution+argmax
        is_zero = x == 0
        gap = next((
            i for i in range(len(x)-1-self.zero_gap, 0, -1) 
            if is_zero[i:i+self.zero_gap].all()
        ), None)
        x = x[gap:]

        # pad to correct length
        if self.input_length:
            n = self.input_length - len(x)
            x = np.pad(x, ((n, 0),)) if n > 0 else x[-n:]
        return x[None, :, None]

    def forward(self, x):
        y, = self.model.run(None, {self.input_name: x})
        y = y[:, :, 1]
        return y
    
    def agg(self, y):
        if self.mean_window:
            return np.minimum(y[:, -1], y[:, -self.mean_window:-1].mean(1))
        return y[:, -1]




class MLFilter(FloodDetector):
    def _filter(self, msg, depth, t):
        # get inputs and pass to model
        x = [d for _, d, _ in self.buffer]
        x = self.preprocess(x)
        y = self.forward(x)

        

        # post-process outputs
        detected = min(np.mean(y[-self.mean_window:-1]), y[-1]) if self.mean_window else y[-1]
        msg['flood_detected'] = detected
        yield msg, t