## WARNING: Do not edit the file directly. It is generated from las_dataset_generatot.ipyn !!!!

import pyproj
import rasterio
import numpy as np
import scipy.ndimage
from scipy.ndimage import zoom
from math import cos, sin, radians, atan2, degrees
from collections import namedtuple
import skimage.transform 

Predictions = namedtuple('Predictions', ['x', 'y', 'label', 'dx', 'dy', 'angle'])
Sample = namedtuple('Sample', ['data', 'predicted'])

EPSG2223 = pyproj.Proj(init="epsg:2223", preserve_units=True)
EPSG26949 = pyproj.Proj(init="epsg:26949", preserve_units=True)
        
def normalize_angle(angle):
    angle = radians(angle)
    angle = atan2(sin(angle), cos(angle))
    angle = degrees(angle)
    return angle
    
class DataProvider(object):
    def __init__(self, radius_in_pixels=32, jitter=0.25, split_ratio=.1):        
        self.densities_path = '/home/femianjc/shared/srp/try2/stack.vrt'
        #self.colors_path = '/home/femianjc/shared/srp/rgb/rgb.vrt'
        self.colors_path = '/home/femianjc/Projects/SRP/transformer_box/sec11-26949.tif'
        self.sample_path = '/home/femianjc/shared/srp/sample_locations_epsg26949.npz'
        self.radius_in_pixels = radius_in_pixels
        self.jitter = jitter
        self.split_ratio = split_ratio
        self._open_datasets()
        
      
    def _open_datasets(self):
        self.densities = rasterio.open(self.densities_path)
        self.colors = rasterio.open(self.colors_path)
        samples= np.load(self.sample_path)
        self.pos_xy = samples['pos_xy']
        self.neg_xy = samples['neg_xy']
        self.pos_angles = samples['pos_angles']
        self.train_pos = np.arange(len(self.pos_xy))
        self.train_neg = np.arange(len(self.pos_xy))
        np.random.shuffle(self.train_pos)
        np.random.shuffle(self.train_neg)
        k_pos = int(self.split_ratio * len(self.train_pos))
        k_neg = int(self.split_ratio * len(self.train_neg))
        self.test_pos = self.train_pos[:k_pos]
        self.train_pos = self.train_pos[k_pos:]
        self.test_neg = self.train_pos[:k_neg]
        self.train_neg = self.train_pos[k_neg:]

    
    def random_positive(self, jitter=None, rotate=True, train=True):
        jitter = self.jitter if jitter is None else jitter
        if train:
            index = np.random.choice(self.train_pos)
        else:
            index = np.random.choice(self.test_pos)
#         print "generated positive, index={}".format(index)
        dx, dy = jitter*np.random.randn(2)
        x, y = self.pos_xy[index] 
        x -= dx
        y -= dy
        label=1
        if rotate:
            rotation = np.random.randint(0, 360)
            patch = self.get_patch_xyr(x, y, rotation)        
            angle = normalize_angle(rotation + self.pos_angles[index])
        else:
            patch = self.get_patch_xy(x, y)
            angle = self.pos_angles[index]
            
        return Sample(patch, Predictions(x, y, label, dx, dy, angle))
    
    def random_negative(self, jitter=None, rotate=True, train=True):
        jitter = self.jitter if jitter is None else jitter
        if train:
            index = np.random.choice(self.train_neg)
        else:
            index = np.random.choice(self.test_neg)
#         print "generated negative, index={}".format(index)
        dx, dy = jitter*np.random.randn(2)
        x, y = self.neg_xy[index] 
        x -= dx
        y -= dy
        label = 0
        
        if rotate:
            rotation = np.random.randint(0, 360)
            patch = self.get_patch_xyr(x, y, rotation)        
            angle = normalize_angle(rotation)
        else:
            patch = self.get_patch_xy(x, y)
            angle = 0
            
        return Sample(patch, Predictions(x, y, label, dx, dy, angle))
    
    def random_sample(self, jitter=None, rotate=True, prob_positive=0.5, train=True):
        jitter = self.jitter if jitter is None else jitter
        if np.random.rand() > prob_positive:
            return self.random_negative(jitter, rotate, train=train)
        else:
            return self.random_positive(jitter, rotate, train=train)
        
    def get_patch_xyr(self, x, y, angle, radius_in_pixels=None):
        radius_in_pixels = radius_in_pixels or self.radius_in_pixels
        
        source_patch = self.get_patch_xy(x, y, radius_in_pixels*2)
#         width = height = 2*radius_in_pixels
                
#         radians = np.radians(angle)
#         c, s = cos(radians), sin(radians)
#         R = np.matrix([[c, -s], 
#                        [s, c]])
#         X = np.asarray([width, height])
#         X = np.asarray(X-R.dot(X)).flatten()
        
#         rotated_patch = np.empty_like(source_patch)
#         for i in range(len(rotated_patch)):
        rotated_patch = source_patch.copy()
        for i in range(6):
            rotated_patch[i] = skimage.transform.rotate(source_patch[i], angle,
                                                
                                                        preserve_range=True)
    
#             scipy.ndimage.affine_transform(source_patch[i],
#                                            matrix=R, offset=X, 
#                                            output_shape = rotated_patch[i].shape,
#                                            output=rotated_patch[i])
        
#         x, y = int((source_patch.shape[2]-width)/2), int( (source_patch.shape[1]-height)/2)
        R = radius_in_pixels
        cropped_patch = rotated_patch[:, R:, R:][:,:2*R, :2*R].copy()
        return cropped_patch    
        
    
    def get_patch_xy(self, x, y, radius_in_pixels=None):
        radius_in_pixels = radius_in_pixels or self.radius_in_pixels
        R = radius_in_pixels
        #x_2223, y_2223 = pyproj.transform(EPSG26949, EPSG2223, x, y)
        #c_2223, r_2223 = np.asarray(~self.colors.affine * (x_2223, y_2223)).astype(int)
        
        c, r = np.asarray(~self.densities.affine*(x, y)).astype(int)
        window = ((r-R, r+R), (c-R, c+R))
        bounds = self.densities.window_bounds(window)        
        colors = self.colors.read((1,2,3), 
                                  window=self.colors.window(*bounds, boundless=True), 
                                  boundless=True,
                                  out=np.empty((3, 2*R,  2*R), dtype=np.uint8)).astype(np.float32)/255.
        
        densities = self.densities.read(window=window, 
                                        boundless=True, 
                                        out=np.empty((3,  2*R,  2*R), dtype=np.uint16)).astype(np.float32)
        
        combined = np.concatenate([colors, densities])
        
        return combined
    