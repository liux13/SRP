
import sys, os
import caffe
import rasterio
import numpy as np
import skimage.util
import data_provider

from fake_positive_data import fake_positive_data

gtp = data_provider.DataProvider()

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
        self.translate = params.get('translate', 0.5)   # Random translation, given a scale in meters
        self.phase = params.get('phase', 'TRAIN')
        print "I was! I was set up!"
        
        self.synth = 1
        self.sscale = 40
        self.bwidth = 15
        self.edge_factor=0.7
        self.btop_amount = 0.3
    
    def reshape(self,bottom,top):
        # no "bottom"s for input layer
        if len(bottom)>0:
            raise Exception('cannot have bottoms for input layer')
    
        assert len(top) == 5
        top[0].reshape(self.num, 6, self.size, self.size) # reshape the outputs to the proper sizes       
        top[1].reshape(self.num) # Present or not (box=1, not-box=0)
        top[2].reshape(self.num) # dx, dy
        top[3].reshape(self.num) # dx, dy
        top[4].reshape(self.num) # angle (0...90 deg - by 5)
        
        
        # print "I was reshaped, I now have shape ", tuple(top[0].shape)
        
    def forward(self,bottom,top): 
        for i in range (self.num):
            if np.random.rand() < self.synth:
                # print 'We\'re using fake data!!!'
                # make synth
                data = np.zeros((6, 64, 64))
                data[3], angle, dx, dy = fake_positive_data(edge_factor=1, bwidth= self.bwidth)
                data[4], _, _, _= fake_positive_data(angle=angle, txy=(dx, dy), 
                                                     edge_factor=self.edge_factor,
                                                     bwidth= self.bwidth)
                # data[5], _, _, _= fake_positive_data(angle=angle, txy=(dx, dy), edge_factor=0.4)
                data[4] *= self.btop_amount
                data *= self.sscale
                
                top[0].data[i, :, : ,:] = data # 6 bands we need to fix that
                top[1].data[i] = 1    
                top[2].data[i] = np.clip(dx + 32, 0, 63)
                top[3].data[i] = np.clip(dy + 32, 0, 63)
                angle = int(round(angle/5.)) % 18
                top[4].data[i] = angle               
            else:
                sample = gtp.random_sample(rotate=self.rotate, jitter=self.translate, train=(self.phase=='TRAIN'))

                top[0].data[i, :, : ,:] = sample.data
                top[1].data[i] = sample.predicted.label
                if sample.predicted.label:
                    top[2].data[i] = np.clip(sample.predicted.dx + 32, 0, 63)
                    top[3].data[i] = np.clip(-sample.predicted.dy + 32, 0, 63)
                    angle = sample.predicted.angle
                    angle = int(round(angle/5.)) % 18
                    top[4].data[i] = angle
                else:
                    top[2].data[i] = 32
                    top[3].data[i] = 32  # actually 0
                    top[4].data[i] = 0

        

    def backward(self, top, propagate_down, bottom):
        # no back-prop for input layers
        pass
    

class MaskoutLayer(caffe.Layer):
    """
    Fill the ouputs with predefiend values whenever the last bottom is true.
    
    Bottom[0] ==> Top[0]
    Bottom[1] is the selector
    """

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs.")
        
        params = eval(self.param_str)
        self.cval = params['cval']
        self.mask = None
        self.count = 0

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].num != bottom[1].num:
            raise Exception("Inputs must have the same num.")    
        top[0].reshape(*bottom[0].data.shape)

    def forward(self, bottom, top):
        """Replace masked-out values by their default/cval, to avoid having them count as loss"""
        
        top[0].data[...] = bottom[0].data[...]
        self.mask = bottom[1].data[:]==0
        self.count =  np.count_nonzero(self.mask)       
        top[0].data[self.mask,...] = np.tile(self.cval, (self.count,1)).reshape(top[0].data[self.mask,...].shape)

    def backward(self, top, propagate_down, bottom):
        """Set the derivative of masked-out items to 0"""
        bottom[0].diff[...] = top[0].diff[...]
        bottom[0].diff[self.mask, :] = 0