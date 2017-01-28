
from collections import namedtuple
import os
import sys
import numpy as np

# Import caffe and complain loudly if it is misconfigured
try:
    import caffe
except:
    raise ImportError("Caffe not found -- you need to put 'caffe/distribute/python/' in the PYTHONPATH or in `sys.path`")


Result = namedtuple('Result', ['label', 'probs'])

class BoxClassifier(object):
    def __init__(self, use_gpu=True, device=0):
        super(BoxClassifier, self).__init__()
        self.use_gpu = use_gpu          
        self.device = device
        caffe.set_device(device)
       
        if use_gpu:
            caffe.set_mode_gpu() 

        # Load the one and only CAFFE classifier for boxes
        root = os.path.dirname(__file__)
        self.net = caffe.Net(os.path.join(root, 'lenet.prototxt'), 
                             os.path.join(root, 'snapshots_highres/_iter_214581.caffemodel'),
                             caffe.TEST)
    def classify(self, boxes):
        if len(boxes.shape) == 3:
            boxes = np.array([boxes])
        
        if self.net.blobs['data'].num != len(boxes):
            self.net.blobs['data'].reshape(*boxes.shape)
            self.net.reshape()
        probs = self.net.forward(data=boxes)['prob']
        label = probs.argmax(1)
        return Result(label=label, probs=probs)
                             
    def __call__(self, boxes):
        return self.classify(boxes) 