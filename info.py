import sys
import numpy as np

def minmax(list_):
    min_ = +sys.float_info.max
    max_ = -sys.float_info.max
    for x in list_:
        if min_ > x:
            min_=x
        if max_ < x:
            max_ = x
    return min_, max_

def plane(w,b,rangeX,rangeY):
    x_plane = np.arange(*rangeX, 0.1)
    y_plane = np.arange(*rangeY, 0.1)
    xx_plane, yy_plane = np.meshgrid(x_plane, y_plane)
    zz_plane = -(w[0]*xx_plane + w[1]*yy_plane + b) / w[2]
    return xx_plane, yy_plane, zz_plane

