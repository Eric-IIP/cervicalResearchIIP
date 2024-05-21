import cv2
import numpy as np

def myfunc(image, label):
    """
    予測画像とラベル画像の比較をし、precision、recall、F値と
    差分画像を出力します
    
    :param image: 予測画像
    :param label: ラベル画像
    
    :return: precision, recall, F値, 画像
    """
    ysize, xsize = label.shape[:2]
    _ysize, _xsize = image.shape[:2]
    if (ysize != _ysize) or (xsize != _xsize):
        image = cv2.resize(image, (xsize, ysize), interpolation=cv2.INTER_CUBIC)

    # 大津の二値化
    th, image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)

    # 閾値による二値化
    threshold = 1
    # th, image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

    eval_image = np.zeros((ysize, xsize,3), dtype=np.uint8)
    tp = fp = fn = tn = 0.00000001
    for y in range(0,ysize):
        for x in range(0,xsize):
            if label[y][x] == image[y][x]:
                if label[y][x] == 255:
                    tp = tp + 1
                    eval_image[y][x][1] = 255
                else:
                    tn = tn + 1
            else:
                if label[y][x] == 255:
                    fn = fn + 1
                    eval_image[y][x][0] = 255
                else:
                    fp = fp + 1
                    eval_image[y][x][2] = 255

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    fMesure = 2 * precision * recall / (precision + recall)
    
    return precision, recall, fMesure, eval_image

