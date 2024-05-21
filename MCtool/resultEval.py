import cv2
import numpy as np


def compare_res_lab(image, label):
    """
    予測画像とラベル画像の比較をし、precision、recall、F値と
    差分画像を出力します
    
        :param: mage: (後処理を施した)予測画像
        :param: abel: ラベル画像
        
        :returns: precision, recall, F値, (TP/FP/FNの色分けをした)画像
    """
    ysize, xsize = label.shape[:2]
    _ysize, _xsize = image.shape[:2]
    if (ysize != _ysize) or (xsize != _xsize):
        image = cv2.resize(image, (xsize, ysize), interpolation=cv2.INTER_CUBIC)

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


def overlay_eval(image, oriimage, label):
    """
    予測画像とテスト画像を重ね合わせた画像を出力する
    
    :param image: (後処理を施した)予測画像
    :param orti_image: 原画像
    :param label: ラベル画像
    
    :returns:  予測画像とテスト画像を重ね合わせた画像
    """
    ysize, xsize = label.shape[:2]
    _ysize, _xsize = image.shape[:2]
    if (ysize != _ysize) or (xsize != _xsize):
        image = cv2.resize(image, (xsize, ysize), interpolation=cv2.INTER_CUBIC)

    # 閾値による二値化
    threshold = 1
    # th, image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

    eval_image = np.zeros((ysize, xsize,3), dtype=np.uint8)

    for y in range(0,ysize):
        for x in range(0,xsize):
            if label[y][x] == image[y][x]:
                if label[y][x] == 255:
                    eval_image[y][x][1] = 255
                else:
                    eval_image[y][x][0] = oriimage[y][x]
                    eval_image[y][x][1] = oriimage[y][x]
                    eval_image[y][x][2] = oriimage[y][x]
            else:
                if label[y][x] == 255:
                    eval_image[y][x][0] = 255
                else:
                    eval_image[y][x][2] = 255

    return eval_image
