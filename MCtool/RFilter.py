from enum import Enum

import cv2
import numpy as np
import math

class FilterKernel: 
    """
    フィルタの行列．ここにはソーベル(SBL)，ラプラシアン(LPL)，エンボス(EMB)がある．
:param  ソーベル規則: SBL[XorY][3or5][3or5]，例) SBLX33はX方向に3×3
:param  ラプラシアン規則: LPL[4or8], LPL4は4近傍，LPL8は8近傍
:param  エンボス規則: EMB[33or35or53], 例)EMB33は3×3
    """
    SBLX33 = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])    
    SBLY33 = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]])
    SBLX35 = np.array([[-1, 0, 0, 0, 1],
                   [-2, 0, 0, 0, 2],
                   [-1, 0, 0 ,0, 1]])
    SBLY53 = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0],
                   [1, 2, 1]])
    SBLX55 = np.array([[-1, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0],
                   [-2, 0, 0, 0, 2],
                   [0, 0, 0, 0, 0],
                   [-1, 0, 0 ,0, 1]])
    SBLY55 = np.array([[-1, 0, -2, 0, -1],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [1, 0, 2, 0, 1]])
    LPL4 = np.array([[0, 1, 0],
                   [1, -4, 1],
                   [0, 1, 0]])
    LPL8 = np.array([[1, 1, 1],
                   [1, -8, 1],
                   [1, 1, 1]]) 
    EMB33 = np.array([[-2, -1, 0],
                   [-1, 1, 1],
                   [0, 1, 2]])  
    EMB35 = np.array([[-2, 0, -1, 0, 0],
                   [-1, 0, 1, 0, 1],
                   [0, 0, 1 ,0, 2]])
    EMB53 = np.array([[-2, -1, 0],
                   [0, 0, 0],
                   [-1, 1, 1],
                   [0, 0, 0],
                   [0, 2, 1]])

class ImgStruct_Name(Enum):
    """
    カーネルの構造の形．
:param  RECT: 矩形
:param  CIRCLE: 円形
:param  CROSS: 十字型(現時点使用していない)
    """
    RECT = 0
    CIRCLE = 1
    CROSS = 2

class ImgStruct_Set:
    """
    カーネルの座標集合
    :examples:
    とある大きさ(img_h,img_w)の二次元画像から，
    座標(x,y)中心，ksizeのkernel_shapeに含まれる座標集合を取得したいとき，
    ### ImgStruct_Set(x,y,img_h,img_w).get(kernel_shape, ksize)
    """
    # コンストラクタ
    def __init__(self, x, y, img_size):
        self.x = x
        self.y = y
        self.img_size = img_size
    
    # (x,y)中心のカーネル内にある座標集合
    def get(self, str_name, ksize=1):
        kernel_set = {(self.x, self.y)}
        height, width = self.img_size
        # ksize×ksizeの矩形カーネルに当てはまる座標
        if str_name == ImgStruct_Name.RECT:
            for i in range(-ksize, ksize+1):
                if 0 <= (self.x + i) and (self.x + i) < width:
                    for j in range(-ksize, ksize+1):
                        if 0 <= (self.y + j) and (self.y + j) < height:
                            kernel_set.add((self.x+i, self.y+j))
        # 半径ksizeの円形カーネルに当てはまる座標
        elif str_name == ImgStruct_Name.CIRCLE:
            for i in range(-ksize, ksize+1):
                if 0 <= (self.x + i) and (self.x + i) < width:
                    for j in range(-ksize, ksize+1):
                        if 0 <= (self.y + j) and (self.y + j) < height:
                            if i*i + j*j <=  ksize*ksize:
                                kernel_set.add((self.x+i, self.y+j))
        # 半径ksizeの十字型カーネルに当てはまる座標(デフォルト:４近傍)
        elif str_name == ImgStruct_Name.CROSS:
            for i in range(-ksize, ksize+1):
                if 0 <= (self.x + i) and (self.x + i) < width:
                    kernel_set.add((self.x+i, self.y))
            for j in range(-ksize, ksize+1):
                if 0 <= (self.y + j) and (self.y + j) < height:
                    kernel_set.add((self.x, self.y+j))
        
        return kernel_set

def gray(image):
    """
    型変換.あとで要確認
    """
    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape
        
    image = np.asarray(image, dtype=np.uint8).reshape((height, width))
    return image

def normalization(image, s):
    """
    画像を[-s*σ, s*σ]から[0,255]に正規化します．
    
    :param  image: 入力画像 
    :param  s: 取る領域(1 or 2 or 3)
    
    :return:  f_img: 正規化した画像
    """
    if len(image.shape) == 2:
        height, width = image.shape
        r_image = np.reshape(image, (height, width, 1))
    else: 
        r_image = image
    
    # 画像サイズの取得
    height, width, deepth = r_image.shape
    # 結果画像 
    f_img = np.zeros((height, width, deepth))

    # 正規化
    for z in range(0,deepth):
        z_image = r_image[:,:,z]
        # zチャンネルの一次元配列化
        img = np.array(z_image).flatten()
        # 平均と分散の算出
        mean = img.mean()
        std = np.std(img)

        # zチャンネルでの正規化
        for y in range(0,height):
            for x in range(0,width):
                if std < pow(10,-10):
                    f_img[y][x][z] = 127
                else:
                    value = round((z_image[y][x] - mean + s*std) / (2*s*std) * 255)
                    if value > 255:
                        f_img[y][x][z] = 255
                    elif value < 0:
                        f_img[y][x][z] = 0
                    else:
                        f_img[y][x][z] = value
        
    # f_img = np.asarray(f_img, dtype=np.float64).reshape((height, width))
    return f_img
    
def list_median(value_list):
    """
    数値のリストから中央値を取得
    """
    med_i = round(len(value_list)/2)
    sort_list = sorted(value_list)
    if len(sort_list) % 2 == 1:
        return int(sort_list[med_i])
    else:
        return (int(sort_list[med_i-1]) + int(sort_list[med_i])) / 2

def median(image, kernel_shape=ImgStruct_Name.CIRCLE, ksize=4, nml_on=False):
    """
    画像にメディアンフィルタを適用する．
    
    :param  image: 入力画像
    :param  kernel_shape: カーネルの構造． ImgStruct_Name.CIRCLE:円形，ImgStruct_Name:矩形
    :param  ksize: カーネルサイズ(円の半径) 
    :param  nml_on: 正規化をかける(デフォルト: NML2をかけない！！！)

    :return:  f_img: メディアンフィルタをかけてからトップハット変換した画像
    """
    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape
        
    image = np.asarray(image, dtype=np.uint8).reshape((height, width))
    f_img = np.zeros((height, width))
    for y in range(0,height):
        for x in range(0,width):
            kernel_set = ImgStruct_Set(x,y,image.shape).get(kernel_shape, ksize)
            value_list = []
            for coor in kernel_set:
                coor_x, coor_y = coor
                value_list.append(image[coor_y][coor_x])
            med = list_median(value_list)
                
            f_img[y][x] = round(med)
    if nml_on:
        f_img = normalization(f_img, 2)
    # f_img = np.asarray(f_img, dtype=np.float64).reshape((height, width))
    return f_img

def tophat(image, kernel_shape=cv2.MORPH_ELLIPSE, ksize=(4,4), nml_on=True):
    """
    画像をトップハット変換します．
    
    :param  image: 入力画像  
    :param  shape: カーネルの構造．cv2のcv2.MORPH_ELLIPSE:楕円形(デフォルト)，cv2.MORPH_RECT:矩形   
    :param  ksize: カーネルサイズ．タプル，カーネルの構造に合わせて入力(デフォルト: 楕円形，(4,4))
    :param  nml_on: 正規化をかける(デフォルト: NML2をかける)

    :return:  f_img: トップハット変換した画像    
    """
    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape
        
    image = np.asarray(image, dtype=np.uint8).reshape((height, width))
    kernel = cv2.getStructuringElement(kernel_shape, ksize)
    f_img = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    if nml_on:
        f_img = normalization(f_img, 2)
    # f_img = np.asarray(f_img, dtype=np.float64).reshape((height, width))
    return f_img

def median_tophat(image, median_ksize, tophat_ksize, nml_on=True):
    """
    画像にメディアンフィルタをかけてからトップハット変換を行う．
    このとき用いるカーネルの構造はメディアンフィルタもトップハット変換も円形
    
    :param  image: 入力画像
    :param  media_ksize: カーネルサイズ(円の半径)
    :param  tophat_ksize: カーネルサイズ(円の半径)
    :param  nml_on: 正規化をかける(デフォルト: NML2をかける)

    :return:  f_img: メディアンフィルタをかけてからトップハット変換した画像
    """
    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape
        
    image = np.asarray(image, dtype=np.uint8).reshape((height, width))
    f_m_img = median(image, ksize=median_ksize)
    f_img = tophat(f_m_img, ksize=(tophat_ksize,tophat_ksize), nml_on=False)
    if nml_on:
        f_img = normalization(f_img, 2)
    
    # f_img = np.asarray(f_img, dtype=np.float64).reshape((height, width))
    return f_img

def sobel(image, kernel, nml_on):
    """
    ソーベルフィルタを適用する．
    負の勾配も考慮するため，出力されるデータの型はcv2.CV_64Fである．
    
    :param  image: 入力画像
    :param  kernel: カーネル．FilterKernel内のカーネルを選択(SBL○××)．
    :param  nml_on: 正規化をかける(デフォルト: NML2をかける)

    :return:  f_img: ソーベルフィルタを適用した画像
    """
    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape
        
    image = np.asarray(image, dtype=np.uint8).reshape((height, width))
    f_img = cv2.filter2D(image, -1, kernel)
    if nml_on:
        f_img = normalization(f_img, 2)
    # f_img = np.asarray(f_img, dtype=np.float64).reshape((height, width))
    return f_img

def sobel_mag(image, nml_on=True):
    """
    ソーベルフィルタマグニチュードを適用する．(M = sqrt(dx^2+dy^2))
    カーネルはx方向，y方向どちらも3×3のフィルタを使用する
    
    :param  image: 入力画像  
    :param  nml_on: 正規化をかける(デフォルト: NML2をかける)

    :return:  f_img: ソーベルフィルタを適用した画像
    """
    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape
        
    image = np.asarray(image, dtype=np.uint8).reshape((height, width))
    dx_img = sobel(image, FilterKernel.SBLX33, nml_on=False)
    dx_img = np.asarray(dx_img, dtype=np.float64).reshape((height, width, 1))
    dy_img = sobel(image, FilterKernel.SBLY33, nml_on=False)
    dy_img = np.asarray(dy_img, dtype=np.float64).reshape((height, width, 1))
    f_img = cv2.magnitude(dx_img, dy_img)
    if nml_on:
        f_img = normalization(f_img, 2)
    # f_img = np.asarray(f_img, dtype=np.float64).reshape((height, width))
    return f_img

def sobel_dir(image, nml_on=True):
    """
    ソーベルディレクションを適用する．(M = {255(arctan(dy/dx)+π)}/{2π})
    カーネルはx方向，y方向どちらも3×3のフィルタを使用する
    
    :param  image: 入力画像  
    :param  nml_on: 正規化をかける(デフォルト: NML2をかける)

    :return:  f_img: ソーベルフィルタを適用した画像
    """
    dx_img = sobel(image, FilterKernel.SBLX33, nml_on=False)
    dy_img = sobel(image, FilterKernel.SBLY33, nml_on=False)
    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape
        
    image = np.asarray(image, dtype=np.uint8).reshape((height, width))
    f_img = np.zeros((height, width))
    for y in range(0,height):
        for x in range(0,width):
            dx = dx_img[y][x]
            dy = dy_img[y][x]
            value = (math.atan2(dy, dx) + math.pi) * 255 / (2 * math.pi)
            f_img[y][x] = round(value)
    if nml_on:
        f_img = normalization(f_img, 2)
    # f_img = np.asarray(f_img, dtype=np.float64).reshape((height, width))
    return f_img

def laplacian(image, kernel, nml_on=True):
    """
    ラプラシアンフィルタを適用する．
    出力されるデータの型はcv2.CV_64Fである．
    
    :param  image: 入力画像
    :param  kernel: カーネル．FilterKernel内のLPL4(4近傍)またはLPL8(8近傍)
    :param  nml_on: 正規化をかける(デフォルト: NML2をかける)

    :return:  f_img: ラプラシアンフィルタを適用した画像
    """
    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape
        
    image = np.asarray(image, dtype=np.uint8).reshape((height, width))
    f_img = cv2.filter2D(image, -1, kernel)
    if nml_on:
        f_img = normalization(f_img, 2)
    # f_img = np.asarray(f_img, dtype=np.float64).reshape((height, width))
    return f_img

def mean(image, ksize=(3,3), nml_on=True):
    """
    平均化フィルタを適用する．
    
    :param  image: 入力画像
    :param  ksize: カーネルサイズ.タプル，デフォルト(3,3)
    :param  nml_on: 正規化をかける(デフォルト: NML2をかける)

    :return:  f_img: 平均化フィルタを適用した画像
    """
    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape
        
    image = np.asarray(image, dtype=np.uint8).reshape((height, width))
    f_img = cv2.blur(image, ksize)
    if nml_on:
        f_img = normalization(f_img, 2)
    # f_img = np.asarray(f_img, dtype=np.float64).reshape((height, width))
    return f_img

def gaussian(image, ksize, nml_on=True):
    """
    ガウシアンフィルタを適用する．
    
    :param  image: 入力画像
    :param  ksize: カーネルサイズ.3or5
    :param  nml_on: 正規化をかける(デフォルト: NML2をかける)

    :return:  f_img: ガウシアンフィルタを適用した画像
    """
    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape
        
    image = np.asarray(image, dtype=np.uint8).reshape((height, width))
    kernel = np.zeros((ksize,ksize))
    sigma = 0.3 * (int(ksize/2) - 1) + 0.8
    for y in range(int(-ksize/2), int(ksize/2)+1):
        for x in range(int(-ksize/2), int(ksize/2)+1):
            kernel[y + int(ksize/2)][x + int(ksize/2)] = \
                math.exp((-x*x-y*y)/(2*sigma*sigma)) / (2 * math.pi*sigma*sigma)
    f_img = cv2.filter2D(image, -1, kernel)
    if nml_on:
        f_img = normalization(f_img, 2)
    # f_img = np.asarray(f_img, dtype=np.float64).reshape((height, width))
    return f_img

def local_binary_pattern(image, p=1, nml_on=True):
    """
    ローカルバイナリパターンを適用する．
    
    :param  image: 入力画像
    :param  p: フィルタの種類．1: 3×3， 2: 5×5(一個飛ばし)
    :param  nml_on: 正規化をかける(デフォルト: NML2をかける)

    :return: f_img: ローカルバイナリパターンを適用した画像
    """
    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape
        
    image = np.asarray(image, dtype=np.uint8).reshape((height, width))
    f_img = np.zeros((height, width))
    for y in range(1, height-p):
        for x in range(1, width-p):
            center_value = image[y][x]
            bp = np.zeros((2*p+1,2*p+1))
            for dy in range(-p, p+1, p):
                for dx in range(-p, p+1, p):
                    value = image[y+dy][x+dx]
                    if value > center_value:
                        bp[dx+p][dy+p] = 1
                    else:
                        bp[dx+p][dy+p] = 0
            value = bp[0][0] * 128 + bp[1*p][0] * 64 + bp[2*p][0] * 32\
                    + bp[2*p][1*p] * 16 + bp[2*p][2*p] * 8\
                    + bp[1*p][2*p] * 4 + bp[0][2*p] * 2 + bp[0][1*p]
            f_img[y][x] = value
    if nml_on:
        f_img = normalization(f_img, 2)
    # f_img = np.asarray(f_img, dtype=np.float64).reshape((height, width))
    return f_img

def median_lbp(image, median_ksize=1, lbp_p=1, nml_on=True):
    """
    画像にメディアンフィルタをかけてからローカルバイナリパターンを行う．
    このとき用いるカーネルの構造はメディアンフィルタは円形
    
    :param  image: 入力画像
    :param  media_ksize: カーネルサイズ(円の半径)
    :param  lbp_p: ローカルバイナリパターンのフィルタの種類
    :param  nml_on: 正規化をかける(デフォルト: NML2をかける)

    :return:  f_img: メディアンフィルタをかけてからローカルバイナリパターン
    """
    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape
        
    image = np.asarray(image, dtype=np.uint8).reshape((height, width))
    f_m_img = median(image, ksize=median_ksize)
    f_img = local_binary_pattern(f_m_img, lbp_p, nml_on=False)
    if nml_on:
        f_img = normalization(f_img, 2)
    # f_img = np.asarray(f_img, dtype=np.float64).reshape((height, width))
    return f_img

def exponential_tone_curve(image, gamma, nml_on=True):
    """
    指数対数型トーンカーブ
    
    :param  image: 入力画像
    :param  gamma: 255*(x/255)^(1/gammma)のパラメータ(現在2.0or0.5)
    :param  nml_on: 正規化をかける(デフォルト: NML2をかける)

    :return:  f_img: 指数対数型トーンカーブをかけた画像
    """
    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape
        
    image = np.asarray(image, dtype=np.uint8).reshape((height, width))

    f_img = np.zeros((height, width))
    for y in range(0, height):
        for x in range(0, width):
            value = 255*math.pow(image[y][x]/255, 1/gamma)
            f_img[y][x] = round(value)
    if nml_on:
        f_img = normalization(f_img, 2)
    # f_img = np.asarray(f_img, dtype=np.float64).reshape((height, width))
    return f_img

def sigmoid_tone_curve(image, b, c, nml_on=True):
    """
    S字型トーンカーブ
    
    :param  image: 入力画像
    :param  b, c: 255*sin((x/255 - b)π) + 1) / c のパラメータ(現在(b,c)=(0.0,2.0)or(1.0,2.0))
    :param  nml_on: 正規化をかける(デフォルト: NML2をかける)

    :return:  f_img: S字型トーンカーブをかけた画像
    """
    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape
        
    image = np.asarray(image, dtype=np.uint8).reshape((height, width))
    f_img = np.zeros((height, width))
    for y in range(0, height):
        for x in range(0, width):
            value = 255*(math.sin(math.pi*(image[y][x]/255 - b)) + 1) / c
            f_img[y][x] = round(value)
    if nml_on:
        f_img = normalization(f_img, 2)
    # f_img = np.asarray(f_img, dtype=np.float64).reshape((height, width))
    return f_img

def hist_flat(image, nml_on=True):
    """
    ヒストグラム平坦化
    
    :param  image: 入力画像
    :param  nml_on: 正規化をかける(デフォルト: NML2をかける)

    :return:  f_img: ヒストグラム平坦化をかけた画像
    """
    # 出力画像の用意
    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape
    image = np.asarray(image, dtype=np.uint8).reshape((height, width))

    f_img = np.zeros((height, width))

    # 画素値LUT
    value_counter = [0] * 256

    # 画素値LUT穴埋め
    for y in range(0, height):
        for x in range(0, width):
            value = image[y][x]
            value_counter[value] = value_counter[value]+1
    
    # 平坦化後の各レベルの画素数を求める
    base_rate = math.ceil(height*width/256)

    # 出力濃度レベルのLUT
    lut = [0] * 256

    # 最小画素数のカウンタ
    min = 0

    # 出力濃度レベル
    level = 0

    # 画素数の累計カウンタ
    pixel_counter = 0

    # 出力濃度レベルのLUT穴埋め    
    for i in range(0, 256):
        pixel_counter = value_counter[i] + pixel_counter
        if pixel_counter >= base_rate:
            for j in range(min, i+1):
                lut[j] = level
            level = int(pixel_counter/base_rate) + level
            min = i + 1
            pixel_counter = pixel_counter % base_rate
        elif i == 255:
            for j in range(min, i+1):
                lut[j] = level
    
    # 平坦化処理
    for y in range(0, height):
        for x in range(0, width):
            f_img[y][x] = lut[image[y][x]]

    if nml_on:
        f_img = normalization(f_img, 2)
    # f_img = np.asarray(f_img, dtype=np.float64).reshape((height, width))
    return f_img

def negaposi(image, nml_on=True):
    """
    ネガポジ反転
    
    :param  image: 入力画像
    :param  nml_on: 正規化をかける(デフォルト: NML2をかける)

    :return:  f_img: ネガポジ反転をかけた画像
    """
    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape
    image = np.asarray(image, dtype=np.uint8).reshape((height, width))
    f_img = cv2.bitwise_not(image)

    if nml_on:
        f_img = normalization(f_img, 2)
    # f_img = np.asarray(f_img, dtype=np.float64).reshape((height, width))
    return f_img


def posterization(image, n, nml_on=True):
    """
    n段階のポスタリゼーション
    
    :param  image: 入力画像
    :param  n: パラメータ(現在10or20or30)
    :param  nml_on: 正規化をかける(デフォルト: NML2をかける)

    :return:  f_img: n段階のポスタリゼーションをかけた画像
    """
    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape
        
    image = np.asarray(image, dtype=np.uint8).reshape((height, width))

    f_img = np.zeros((height, width))
    for y in range(0, height):
        for x in range(0, width):
            n_value = math.floor(image[y][x] * n / 255)
            value = round(255 * n_value / (n-1))
            f_img[y][x] = value
    if nml_on:
        f_img = normalization(f_img, 2)
    # f_img = np.asarray(f_img, dtype=np.float64).reshape((height, width))
    return f_img

def solarization(image, nml_on=True):
    """
    ソラリゼーション
    
    :param  image: 入力画像
    :param  nml_on: 正規化をかける(デフォルト: NML2をかける)

    :return:  f_img: ソラリゼーションをかけた画像
    """
    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape
        
    image = np.asarray(image, dtype=np.uint8).reshape((height, width))
    f_img = np.zeros((height, width))
    for y in range(0, height):
        for x in range(0, width):
            const_value = 2 * math.pi / 255
            value = abs(round(math.sin(image[y][x]*const_value)*255))
            f_img[y][x] = value
    if nml_on:
        f_img = normalization(f_img, 2)
    # f_img = np.asarray(f_img, dtype=np.float64).reshape((height, width))
    return f_img

def emboss(image, kernel, nml_on=True):
    """
    エンボスフィルタを適用する．
    
    :param  image: 入力画像
    :param  kernel: カーネル．FilterKernel内のカーネルを選択(EMB××)．
    :param  nml_on: 正規化をかける(デフォルト: NML2をかける)

    :return:  f_img: ソーベルフィルタを適用した画像
    """
    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape
        
    image = np.asarray(image, dtype=np.uint8).reshape((height, width))
    f_img = cv2.filter2D(image, -1, kernel)
    if nml_on:
        f_img = normalization(f_img, 2)
    # f_img = np.asarray(f_img, dtype=np.float64).reshape((height, width))
    return f_img

def knn_ave(image, ksize, k, nml_on=True):
    """
    最近傍化フィルタ 
    ksize*ksizeの中から注目画素の画素値と近いk個の画素を選択し，その平均を出力する
    
    :param  image: 入力画像
    :param  ksize: 注目領域の半径(現在3or5)
    :param  k: 近傍k個の画素値(現在3or5)
    :param  nml_on: 正規化をかける(デフォルト: NML2をかける)

    :return:  f_img: 最近傍化フィルタをかけた画像
    """
    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape
        
    image = np.asarray(image, dtype=np.uint8).reshape((height, width))
    f_img = np.zeros((height, width))
    for y in range(ksize, height-ksize):
        for x in range(ksize, width-ksize):
            v = image[y][x]
            ii = 0
            value = 0
            value_list = []
            for dy in range(-ksize,ksize+1):
                for dx in range(-ksize,ksize+1):
                    value_list.append(image[y+dy][x+dx])
            sort_list = sorted(value_list)
            for i in range(0, len(value_list)):
                if sort_list[i] >= v:
                    ii = i
                    break
            if ii >= int(k/2) and ii < len(value_list) - int(k/2):
                for i in range(ii-int(k/2), ii+int(k/2)+1):
                    value = sort_list[i] + value 
            elif ii < len(value_list) - int(k/2):
                for i in range(0, k):
                    value = sort_list[i] + value
            else:
                for i in range(len(value_list)-k, len(value_list)):
                    value = sort_list[i] + value
            value = round(value / k)
            f_img[y][x] = value
    if nml_on:
        f_img = normalization(f_img, 2)
    # f_img = np.asarray(f_img, dtype=np.float64).reshape((height, width))
    return f_img

def bilateral(image, ksize, sigma1, sigma2, nml_on=True):
    """
    バイラテラルフィルタ．中心画素からの距離と画素値の差に応じて，
    ガウシアン分布に従う重みをつける平均化フィルタの一種
    
    :param  image: 入力画像
    :param  ksize: 注目領域の半径(現在3or5)
    :param  simga1, sigam2: パラメータ(現在(sigma1,sigma2)=(1.0,2.0))
    :param  nml_on: 正規化をかける(デフォルト: NML2をかける)

    :return:  f_img: n段階のポスタリゼーションをかけた画像
    """
    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape
        
    image = np.asarray(image, dtype=np.uint8).reshape((height, width))
    f_img = np.zeros((height, width))
    p = int(ksize)
    for y in range(p, height - p):
        for x in range(p, width - p):
            v = int(image[y][x])
            # 分子部分と分母部分それぞれ分けて計算する
            numerator = 0
            denominator = 0

            for dy in range(-p, p+1):
                for dx in range(-p, p+1):
                    vv = int(image[y+dy][x+dx])
                    coef1 = math.exp((-int(dx)*int(dx) -int(dy)*int(dy)) / (2*int(sigma1)*int(sigma1)))
                    coef2 = math.exp((-int((v-vv))*int((v-vv))) / (2*int(sigma2)*int(sigma2)))
                    coef = coef1 * coef2 
                    numerator = coef*vv + numerator
                    denominator = coef + denominator
            
            value = round(numerator/denominator)
            f_img[y][x] = value
    if nml_on:
        f_img = normalization(f_img, 2)
    # f_img = np.asarray(f_img, dtype=np.float64).reshape((height, width))
    return f_img

def black(image): 
    """
    黒い画像を返す
    """
    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape
        
    image = np.asarray(image, dtype=np.uint8).reshape((height, width))
    f_img = np.zeros((height, width))
    # f_img = np.asarray(f_img, dtype=np.float64).reshape((height, width))
    return f_img

#input_path = "../dataset/cropped-original/image/krhaa001_n_N1_from3072.png"
#image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
# nml_image = bilateral(image, 3, 1, 2)
#nml_image = posterization(image, 10)
#output_path = "test.png"
#cv2.imwrite(output_path, nml_image)

###########################################################################
#### additional filters
###########################################################################

def canny(image, threshold1, threshold2):
    
    height, width = image.shape
        
    image = np.asarray(image, dtype=np.uint8).reshape((height, width))
    #chaning the thresholds
    edges = cv2.Canny(image, threshold1, threshold2)

    return edges

def prewitt(image, variation):

    #default kernel
    prewitt_kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
    prewitt_kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)

    #modified  kernel
    prewitt_kernel_x1 = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
    prewitt_kernel_y1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)

    # sharpened kernel
    prewitt_kernel_x2 = np.array([[2, 0, -2], [2, 0, -2], [2, 0, -2]], dtype=np.float32)
    prewitt_kernel_y2 = np.array([[2, 2, 2], [0, 0, 0], [-2, -2, -2]], dtype=np.float32)

    #softened kernel
    prewitt_kernel_x3 = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32) / 2
    prewitt_kernel_y3 = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32) / 2


    if variation == 0:
        prewitt_x = cv2.filter2D(image, -1, prewitt_kernel_x)
        prewitt_y = cv2.filter2D(image, -1, prewitt_kernel_y)
    elif variation == 1:
        prewitt_x = cv2.filter2D(image, -1, prewitt_kernel_x1)
        prewitt_y = cv2.filter2D(image, -1, prewitt_kernel_y1)
    elif variation == 2:
        prewitt_x = cv2.filter2D(image, -1, prewitt_kernel_x2)
        prewitt_y = cv2.filter2D(image, -1, prewitt_kernel_y2)

    elif variation == 3:
        prewitt_x = cv2.filter2D(image, -1, prewitt_kernel_x3)
        prewitt_y = cv2.filter2D(image, -1, prewitt_kernel_y3)

    prewitt_combined = cv2.magnitude(prewitt_x, prewitt_y)

    return prewitt_combined

def unsharp_masking(image, kernel, sigmax, weight1, weight2):
    # kernel variable is for different gaussian kernels
    # sigmax is for modifying gaussian blur as well
    # weight can be modified for sharpening
    blurred_img = cv2.GaussianBlur(image, kernel, sigmax)

    mask = cv2.subtract(image, blurred_img)

    sharpened_img = cv2.addWeighted(image, weight1, blurred_img, weight2, 0)

    return sharpened_img


def fourier(image, filter_type='low_pass', d0=50, params=None):
    # Convert image to grayscale if it's not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = np.copy(image)

    # Perform FFT
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    # Frequency spectrum
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    # Constructing the filter
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2

    # Determine the type of filter and apply accordingly
    if filter_type == 'low_pass':
        mask = np.zeros((rows, cols), np.uint8)
        mask[crow - d0:crow + d0, ccol - d0:ccol + d0] = 1

    elif filter_type == 'high_pass':
        mask = np.ones((rows, cols), np.uint8)
        mask[crow - d0:crow + d0, ccol - d0:ccol + d0] = 0

    elif filter_type == 'band_pass':
        if params is None or len(params) < 2:
            raise ValueError("Band-pass filter requires parameters 'd0' and 'd1'.")
        d0, d1 = params
        mask = np.zeros((rows, cols), np.uint8)
        mask[crow - d1:crow + d1, ccol - d1:ccol + d1] = 1
        mask[crow - d0:crow + d0, ccol - d0:ccol + d0] = 0

    elif filter_type == 'notch':
        if params is None or len(params) < 2 or not isinstance(params[0], tuple):
            raise ValueError("Notch filter requires parameters 'center' (tuple) and 'radius'.")
        center, radius = params
        mask = np.ones((rows, cols), np.uint8)
        y, x = np.ogrid[:rows, :cols]
        mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius ** 2
        mask[mask_area] = 0

    else:
        raise ValueError(f"Unknown filter type '{filter_type}'.")

    # Apply mask and inverse transform
    fshift = fshift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    #img_back = np.uint8(img_back)
    #img_back = cv2.equalizeHist(img_back)


    return img_back

def erosion(img, variation):
    # default erosion
    if variation == 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        erosion_img = cv2.erode(img, kernel, iterations = 1)

    #smaller elliptical kernel
    elif variation == 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        erosion_img = cv2.erode(img, kernel, iterations = 1)

    #larger elliptical kernel
    elif variation == 2:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        erosion_img = cv2.erode(img, kernel, iterations = 1)

    #rectangular kernel
    elif variation == 3:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        erosion_img = cv2.erode(img, kernel, iterations = 1)

    #cross-shaped kernel
    elif variation == 4:
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))
        erosion_img = cv2.erode(img, kernel, iterations = 1)
    
    #more iterations
    elif variation == 5:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        erosion_img = cv2.erode(img, kernel, iterations = 3)

    return erosion_img

def dilation(img, variation):

    #default dilation
    if variation == 0:
        kernel = np.ones((5, 5), np.uint8)  
        dilated_image = cv2.dilate(img, kernel, iterations=1)
    #smaller kernel
    elif variation == 1:
        kernel = np.ones((3, 3), np.uint8)  
        dilated_image = cv2.dilate(img, kernel, iterations=1)
    #larger kernel
    elif variation == 2:
        kernel = np.ones((7, 7), np.uint8)  
        dilated_image = cv2.dilate(img, kernel, iterations=1)
    # elliptical kernel
    elif variation == 3:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated_image = cv2.dilate(img, kernel, iterations=1)
    # cross kernel
    elif variation == 4:
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
        dilated_image = cv2.dilate(img, kernel, iterations=1)
    # more iterations
    elif variation == 5:
        kernel = np.ones((5, 5), np.uint8)  
        dilated_image = cv2.dilate(img, kernel, iterations=3)


    return dilated_image

def opening(img, variation):

    #default opening filter
    if variation == 0:
        kernel = np.ones((5,5),np.uint8)
        opening  = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    #smaller kernel
    elif variation == 1:
        kernel = np.ones((3,3),np.uint8)
        opening  = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    #bigger kernel
    elif variation == 2:
        kernel = np.ones((7,7),np.uint8)
        opening  = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    #elliptical kernel
    elif variation == 3:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        opening  = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    #cross-shaped kernel
    elif variation == 4:
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))
        opening  = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    return opening

def closing(img, variation):
    if variation == 0:
        kernel = np.ones((5,5),np.uint8)
        closing  = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    #smaller kernel
    elif variation == 1:
        kernel = np.ones((3,3),np.uint8)
        closing  = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    #larger kernel
    elif variation == 2:
        kernel = np.ones((7,7),np.uint8)
        closing  = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    #elliptical kernel
    elif variation == 3:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closing  = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    #cross-shaped kernel
    elif variation == 4:
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
        closing  = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    return closing

# not used due tot the reason that it is basically same as mean filter
def box(img, kernel_size):

    blurred_image = cv2.blur(img, kernel_size)

    return blurred_image

def scharr_grad(img, scale):

    scharr_x = cv2.Scharr(img, cv2.CV_64F, 1, 0) * scale
    scharr_y = cv2.Scharr(img, cv2.CV_64F, 0, 1) * scale

    gradient_magnitude = np.sqrt(scharr_x**2 + scharr_y**2)

    return gradient_magnitude

def roberts_cross(img, variation):
    roberts_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
    roberts_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)

    larger_roberts_x = np.array([[1, 0, 0], [0, -2, 0], [0, 0, 1]], dtype=np.float32)
    larger_roberts_y = np.array([[0, 1, 0], [-1, -2, 1], [0, 0, 0]], dtype=np.float32)

    horizontal_roberts_x = np.array([[1, 0, -1], [0, 0, 0], [0, 0, 0]], dtype=np.float32)
    horizontal_roberts_y = np.array([[0, 1, 0], [0, 0, 0], [0, -1, 0]], dtype=np.float32)

    vertical_roberts_x = np.array([[0, 0, 0], [1, 0, -1], [0, 0, 0]], dtype=np.float32)
    vertical_roberts_y = np.array([[0, 1, 0], [0, 0, 0], [0, -1, 0]], dtype=np.float32)

    if variation == 1:
        # Apply the kernels separately to the image
        roberts_x_filtered = cv2.filter2D(img, -1, roberts_x)
        roberts_y_filtered = cv2.filter2D(img, -1, roberts_y)
    elif variation == 2:
        roberts_x_filtered = cv2.filter2D(img, -1, larger_roberts_x)
        roberts_y_filtered = cv2.filter2D(img, -1, larger_roberts_y)
    elif variation == 3:
        roberts_x_filtered = cv2.filter2D(img, -1, horizontal_roberts_x)
        roberts_y_filtered = cv2.filter2D(img, -1, horizontal_roberts_y)
    elif variation == 4:
        roberts_x_filtered = cv2.filter2D(img, -1, vertical_roberts_x)
        roberts_y_filtered = cv2.filter2D(img, -1, vertical_roberts_y)

    # Compute the magnitude of the gradient
    gradient_magnitude = np.sqrt(roberts_x_filtered**2 + roberts_y_filtered**2)
 
    return gradient_magnitude

#might be same as erode check later
def min(img, kernel_size):

    # Create a custom kernel with all elements set to 1
    kernel = np.ones(kernel_size, np.uint8)

    # Apply the minimum filter using erosion
    min_filtered_image = cv2.erode(img, kernel)
    
    return min_filtered_image

#might be same as dilate
def max(img, kernel_size):

    # Create a custom kernel with all elements set to 1
    kernel = np.ones(kernel_size, np.uint8)

    # Apply the maximum filter using dilation
    max_filtered_image = cv2.dilate(img, kernel)

    return max_filtered_image

def morph_grad(img, kernel_size):

    # Create a custom kernel with all elements set to 1
    kernel = np.ones(kernel_size, np.uint8)

    # Apply the morphological gradient filter
    morphological_gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    
    return morphological_gradient

def morph_laplacian(img, kernel_size):
    kernel = np.ones(kernel_size, np.uint8)

    # Apply morphological dilation
    dilated_image = cv2.dilate(img, kernel)

    # Apply morphological erosion
    eroded_image = cv2.erode(img, kernel)

    # Compute the Laplacian
    laplacian = dilated_image - eroded_image

    # Combine the original image with the Laplacian to enhance edges
    morphological_laplacian = cv2.addWeighted(img, 1, laplacian, -1, 0)

    return morphological_laplacian

def bottom_hat_transform(img, kernel_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

    # Apply the bottom hat transformation
    bottom_hat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

    return bottom_hat

def distance_transform(img):
    _, binary_image = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
    binary_image = cv2.convertScaleAbs(binary_image)
    # Compute the distance transform
    distance_transform = cv2.distanceTransform(binary_image, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    # Normalize the distance transform for visualization
    normalized_distance_transform = cv2.normalize(distance_transform, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    return normalized_distance_transform

def homomorphic(img):
    log_image = np.log1p(np.float32(img))

    # Perform a Fourier transform
    f_transform = np.fft.fft2(log_image)

    # Define the high-pass filter (e.g., Butterworth)
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    D = 30
    n = 2
    mask = np.zeros((rows, cols), np.uint8)
    for i in range(rows):
        for j in range(cols):
            if np.sqrt((i - crow) ** 2 + (j - ccol) ** 2) > D:
                mask[i, j] = 1
    filter_applied = f_transform * (1 - mask)

    # Perform an inverse Fourier transform
    filtered_image = np.fft.ifft2(filter_applied)

    # Exponentiate the filtered image to obtain the final enhanced image
    enhanced_image = np.exp(np.real(filtered_image))

    # Normalize the enhanced image
    enhanced_image = cv2.normalize(enhanced_image, None, 0, 255, cv2.NORM_MINMAX)

    # Convert the enhanced image to uint8
    enhanced_image = np.uint8(enhanced_image)

    return enhanced_image

# bugged
# def structure_tensor(img):
#     gradient_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
#     gradient_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

#     # Compute the structure tensor
#     structure_tensor = np.zeros((img.shape[0], img.shape[1], 2, 2), dtype=np.float32)
#     structure_tensor[:, :, 0, 0] = gradient_x * gradient_x
#     structure_tensor[:, :, 0, 1] = gradient_x * gradient_y
#     structure_tensor[:, :, 1, 0] = gradient_y * gradient_x
#     structure_tensor[:, :, 1, 1] = gradient_y * gradient_y
#     # Reshape structure_tensor to fit cv2.cornerEigenValsAndVecs input requirements
#     structure_tensor_reshaped = structure_tensor.reshape((img.shape[0], img.shape[1], 4))
#     structure_tensor_reshaped = structure_tensor_reshaped.astype(np.float32)

#     # Compute the eigenvalues and eigenvectors of the structure tensor
#     eigenvalues, eigenvectors = cv2.cornerEigenValsAndVecs(structure_tensor_reshaped, blockSize=3, ksize=3)

#     # Perform corner detection using the eigenvalues
#     corners = cv2.cornerHarris(img, blockSize=3, ksize=3, k=0.04)

#     return corners

# the filter unblurs the original img
# blurred in purpose then applied richardson
def richardson_lucy(img):
    blurred_img = cv2.GaussianBlur(img, (5,5), 1.0)
    psf = np.ones((5, 5)) / 25  # Example PSF (5x5 averaging filter)

    # Initialize the estimate of the original image
    estimate = np.copy(blurred_img)

    # Number of iterations
    num_iterations = 10

    # Richardson-Lucy deconvolution
    for _ in range(num_iterations):
        # Estimate the blurred image from the current estimate
        estimated_blurred = cv2.filter2D(estimate, -1, psf)
    
        # Compute the error between the observed blurred image and the estimated blurred image
        error = blurred_img / (estimated_blurred + 1e-8)
    
        # Update the estimate using the error and the transpose of the PSF
        estimate *= cv2.filter2D(error, -1, psf.T)
    
    return estimate

def custom_standard_deviation(img):
    #img = cv2.imread(img)
    
    height, width, channels = img.shape
    overall_value = list()
    for y in height:
        for x in width:
            overall_value.append(img[y][x])
    print()
            
            

