import glob
import io
import cv2
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


def generate_data_format(path1,path2):
        # path1 = "data/train/"
        # path2="data/label/"
        path3="data/train/test/"

        flag=True
        x_train=[]
        y_train=[]

        for i in glob.glob(path1 + '*.png'):
                img = cv2.imread(i)
                y_img = cv2.imread(i.replace(path1, path2,1))
                x_train.append(img)
                y_train.append(y_img)
                
        x_train=np.array(x_train)
        y_train=np.array(y_train)
        # x_train=x_train.reshape((-1,512,512,3))
        x_train = x_train.astype('float32')
        x_train /= 255


        # y_train=y_train.reshape((-1,512,512,3))
        y_train = y_train.astype('float32')
        y_train /= 255


        # cv2.imwrite(path3+'image1.png',x_train[0]*255)
        # cv2.imwrite(path3+'image2.png',x_train[1]*255)

        # cv2.imwrite(path3+'image1_label.png',y_train[0]*255)
        # cv2.imwrite(path3+'image2_label.png',y_train[1]*255)
        return x_train,y_train


        


def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.
     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
