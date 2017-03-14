
import skimage.transform
import skimage.filters
import scipy.ndimage
import numpy as np

def fake_positive_data(angle='random', txy='random', edge_factor=0.4, fg_noise=0.1, bg_noise=0.1, bwidth=20, 
                       sigma=12):
    
    if angle == 'random':
        angle = np.random.randint(0, 89)
    if txy == 'random':
        tx, ty = np.random.randint(-20, 20, 2)
    else:
        tx, ty = txy
    
    square = np.zeros((64, 64))
    square[(64-bwidth)/2:(64+bwidth)/2,(64-bwidth)/2:(64+bwidth)/2] = 1
    
    outline = scipy.ndimage.morphology.morphological_gradient(square, 3)
    outline[(64-bwidth)/2:, (64-bwidth)/2:(64+bwidth)/2] = 0
    square = (1-edge_factor)*square + edge_factor*outline
    
    gradient = np.zeros_like(square)
    gradient[:32] = 1
    gradient = skimage.filters.gaussian(gradient, sigma=sigma)
    square *= gradient
    square /= np.percentile(square.flat, 99.9)

    rotated = skimage.transform.rotate(square, angle, preserve_range=True)
    translation = skimage.transform.AffineTransform(matrix=np.array([[1,0, tx], [0,1,ty], [0,0,1]]))
    translated = skimage.transform.warp(rotated, translation.inverse,  preserve_range=True)

    background = translated == 0
    noisy = translated
    noisy += background*np.random.randn(64, 64)*bg_noise 
    noisy += ~background*np.random.randn(64, 64)*fg_noise
    noisy = noisy.clip(0,1)

    return (noisy, angle, tx, ty)