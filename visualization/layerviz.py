# Adapted from https://github.com/gustavla/caffe-weighted-samples/blob/master/examples/filter_visualization.ipynb

import argparse
import caffe
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Visualize convolutional layers.')
parser.add_argument('-m', '--model', metavar='model', default='training.prototxt', help='Path to the .prototxt file')
parser.add_argument('weights', metavar='weights', nargs='?', help='Path to the .caffemodel file')
parser.add_argument('-b', '--base', metavar='dir', dest='base', nargs=1, help='Base path to work with')

args = parser.parse_args()

if args.base is not None:
    os.chdir(args.base[0])

prototxt = args.model      # 'training.prototxt'
caffemodel = args.weights  # 'snapshots/net_4_iter_38000.caffemodel'
if caffemodel is None:
    snapshots_dir = 'snapshots'
    if not os.path.exists(snapshots_dir):
        print('Unable to find snapshots directory. Please specify a weight file manually.\n')
        parser.print_usage()
        exit(1)

    try:
        newest = max(glob.iglob(os.path.join(snapshots_dir, '*.caffemodel')), key=os.path.getctime)
    except ValueError:
        newest = None
    if newest is None:
        print('Unable to find *.caffemodel file in snapshots directory. Please specify a weight file manually.\n')
        parser.print_usage()
        exit(1)

    print('Using model file: %s' % newest)
    caffemodel = newest

caffe.set_mode_cpu()
net = caffe.Net(prototxt, caffemodel, caffe.TEST)

print('Layer parameters:')
for layer_name, layer in net.params.items():
    print("%30s" % layer_name)
    for param in layer:
        print("%30s %s" % ("->", str(param.data.shape)))

print('Blobs:')
for layer_name, blob in net.blobs.items():
    print("%30s %s" % (layer_name, str(blob.data.shape)))


# take an array of shape (n, height, width) or (n, height, width, channels)
# and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
def vis_square(title, data, padsize=1, padval=1):
    fig = plt.figure(0)
    fig.canvas.set_window_title(title)

    count = data.shape[0]

    # normalize all kernels to 0..1
    data -= data.min()
    data /= data.max()

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(count)))
    padding = ((0, n ** 2 - count), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    # obtain dimensions after padding
    dims = data.shape[1:]  # (height, width, channels)

    # tile the filters into an image
    data = data.reshape((n, n) + dims).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    # flatten single-dimensional entries as the third dimension makes
    # matplotlib expect RGB or RGBA values (i.e three or four channels)
    data = data.squeeze()

    cmap = None
    if data.ndim == 2:
        cmap = 'gray'

    # remove the padding on the right and bottom border
    data = data[0:data.shape[0]-1, 0:data.shape[1]-1]

    # leaving interpolation on gives blurry results for small kernels
    plt.imshow(data, interpolation='none', cmap=cmap)
    plt.show()

title = os.path.basename(caffemodel)

# the parameters are a list of [weights, biases]
layer_params = net.params.itervalues().next()  # net.params['conv1']
weights = layer_params[0]  # weights.data is an array of K MxN arrays of floats for K learned kernels
biases = layer_params[1]   # biases.data is an array of K floats for K learned kernels
kernels = weights.data     # kernels.shape = (count, channels, height, width)
vis_square(title, kernels.transpose(0, 2, 3, 1))
