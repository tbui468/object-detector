import numpy as np
import scipy
import scipy.signal as sig
import imageio
import matplotlib.pyplot as plt
import math


def conv2d(img, kernel, x_axis=True):
    if x_axis:
        kernel = np.flip(kernel, axis=1)
    else:
        kernel = np.flip(kernel, axis=0)

    out = np.zeros(img.shape) 

    padded = np.pad(img, 1)
    extra_rows = (kernel.shape[0] - 1) // 2
    extra_cols = (kernel.shape[1] - 1) // 2

    for row in range(1, padded.shape[0] - 1):
        for col in range(1, padded.shape[1] - 1):
            chunk = padded[row - extra_rows: row + extra_rows + 1, col - extra_cols: col + extra_cols + 1]
            out[row - 1, col - 1] = np.sum(chunk * kernel)

    return out.astype(int)
            
def magnitude(g_x, g_y):
    m = np.sqrt(g_x * g_x + g_y * g_y)
    return m.astype(int)

#returns int between 0 and 180 for HOG
def direction(g_x, g_y):
    d = np.arctan(g_y.astype(float), g_x.astype(float)) + math.pi / 2.0
    d = (d * 180 / math.pi).astype(int)
    return d


img = imageio.imread('manu-2004.jpg', pilmode='L')

#Basic cardinal direction kernel
basic_x = np.array([[-1, 0, 1]])
basic_y = np.array([[1], [0], [-1]])

#Prewitt kernel (8 pixels surrounding pixel)
prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
prewitt_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

#Sobel kernel (8 pixels, but weighted towards pixels in cardinal directions)
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

def test():
    test = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    conv2d(test, sobel_x, x_axis=True)
    print(sig.convolve2d(test, sobel_x, mode='same'))
    conv2d(test, sobel_y, x_axis=False)
    print(sig.convolve2d(test, sobel_y, mode='same'))


def image_gradient_vector():
    #G_x = sig.convolve2d(img, sobel_x, mode='same')
    #G_y = sig.convolve2d(img, sobel_y, mode='same')
    G_x = conv2d(img, sobel_x, x_axis=True)
    G_y = conv2d(img, sobel_y, x_axis=False)

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.imshow((G_x + 255) / 2, cmap='gray'); ax1.set_xlabel('Gx')
    ax2.imshow((G_y + 255) / 2, cmap='gray'); ax2.set_xlabel('Gy')
    plt.show()

def histogram_of_oriented_gradients():
    BUCKET_COUNT = 9
    CELL_SIZE = 8 #each cell is 8x8 pixels
    BLOCK_SIZE = 2 #each block is 2x2 cells

    test = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    G_x = sig.convolve2d(test, sobel_x, mode='same')
    G_y = sig.convolve2d(test, sobel_y, mode='same')

    #split image into cells

    def bucket_cell(cell_gx, cell_gy, bucket_count):
        out = np.zeros(bucket_count)
        mag = magnitude(cell_gx, cell_gy)
        dire = direction(cell_gx, cell_gy)
        partitions = [180 * i // bucket_count for i in range(bucket_count)]

        for row_idx, row in enumerate(mag):
            for col_idx, m in enumerate(row):
                d = dire[row_idx, col_idx]
                m = mag[row_idx, col_idx]
                l_idx = d // 20 
                r_idx = l_idx + 1
                   
                if r_idx == BUCKET_COUNT:
                    out[l_idx] += m
                else:
                    l_ratio = m * (20 - abs(partitions[l_idx] - d)) / 20
                    r_ratio = m * (20 - abs(partitions[r_idx] - d)) / 20
                    out[l_idx] += l_ratio
                    out[r_idx] += r_ratio
        
        return out



    #concat 4 cells, and normalize
    #concat all the normalized blocks, and then feed to SVM (or other classifier) to classify

    print(bucket_cell(G_x[:8, :8], G_y[:8, :8], BUCKET_COUNT))

histogram_of_oriented_gradients()
