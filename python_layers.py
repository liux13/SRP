
import sys, os
import caffe
import rasterio
import numpy as np
import skimage.util
import data_provider

sys.stdout.write('creating gtp')
gtp = data_provider.DataProvider()
sys.stdout.write('it is done')

class BoxInputLayer(caffe.Layer):
    def setup(self,bottom,top):
        # read parameters from `self.param_str`
        params = eval(self.param_str)
        # self.combined_path = params['raster'] # location of a small-ish file with rasterized label into. 
        # self.size = params['size']  # window size (width, height)
        self.num = params.get('num', 8)   # batch size, default is 8
        # self.height_jitter = params.get('height_jitter', 1.0)   # jitter heights (uniform distr.)
        self.rotate = params.get('rotate', True)   # Apply random rotations
        self.size = 2 * gtp.radius_in_pixels
        self.translate = params.get('translate', self.size)   # Apply random rotations
        self.phase = params.get('phase', 'TRAIN')
        print "I was! I was set up!"
    
    def reshape(self,bottom,top):
        # no "bottom"s for input layer
        if len(bottom)>0:
            raise Exception('cannot have bottoms for input layer')
    
        assert len(top) ==len(['data', 'isbox', 'xy', 'angle'])
        top[0].reshape(self.num, 6, self.size, self.size) # reshape the outputs to the proper sizes       
        top[1].reshape(self.num) # Present or not (box=1, not-box=0)
        top[2].reshape(self.num, 2) # dx, dy
        top[3].reshape(self.num, 2) # angle (as cos, sin)
        
        
        # print "I was reshaped, I now have shape ", tuple(top[0].shape)

    def forward(self,bottom,top): 
        sys.stdout.write('D')
        sys.stdout.flush()
        for i in range (self.num):
            sample = gtp.random_sample(rotate=self.rotate, jitter=self.translate, train=(self.phase=='TRAIN'))
            
            top[0].data[i, :, : ,:] = sample.data
            top[1].data[i] = sample.predicted.label
            top[2].data[i, 0] = sample.predicted.dx
            top[2].data[i, 1] = sample.predicted.dy
            top[3].data[i, 0] = np.cos(np.radians(sample.predicted.angle))
            top[3].data[i, 1] = np.sin(np.radians(sample.predicted.angle))
                        
        sys.stdout.write('E')
        sys.stdout.flush()
        

    def backward(self, top, propagate_down, bottom):
        # no back-prop for input layers
        pass
