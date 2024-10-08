import os
import random
from matplotlib import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from Rpkg.Rfund.InputFeature import InputFeature
from Rpkg.Rfund import ReadFile, WriteFile

from MCtool import RFilter
from MCtool.RFilter import FilterKernel

"""
[注意] これはモジュールです．
"""

# パラメータ設定
IMAGE_SIZE = 256
EPOCHS = 100

# 出力パス設定
output_dir = '../experiment/'   # 出力のディレクトリパス

def read_csv_image(rootpath, csvpath):
    """
    csvファイル内に書かれた名前の画像を，
    rootpathのディレクトリから探し，画像のリストを返す．
    
    :param rootpath: 画像が入っているフォルダのパス
    :param csvpath: 画像のファイル名が記述されたcsvファイルのパス
    
    :return: imglist: 画像のリスト
    :return: fnlist: ファイル名のリスト
    """
    imglist = []
    fnlist = []
    csv_file = pd.read_csv(csvpath, header=None)
    
    for i in range(0, len(csv_file)):
        # pngファイルの読み込み
        fn = csv_file[0][i] + '.png'
        # 画像のパス
        imgpath = os.path.join(rootpath, fn)
        # print(imgpath)
        # 画像の読み込み
        img = ReadFile.one_image(imgpath)
        img = np.asarray([img], dtype=np.float64)
        img = np.asarray(img, dtype=np.float64).reshape((IMAGE_SIZE, IMAGE_SIZE, 1))
        # リストへの追加
        imglist.append(img)
        fnlist.append(fn)
    
    return imglist, fnlist

def single_image(img, image_type, nml_on=True):
     # 画像処理を施す
    if image_type == InputFeature.GRY_:
        f_img = RFilter.gray(img) 
        
    elif image_type == InputFeature.NML1:
        f_img = RFilter.normalization(img, 1) 
        
    elif image_type == InputFeature.NML2:
        f_img = RFilter.normalization(img, 2) 
        
    elif image_type == InputFeature.NML3:
        f_img = RFilter.normalization(img, 3) 
        
    elif image_type == InputFeature.TOP1:
        f_img = RFilter.median_tophat(img, 1, 4, nml_on=nml_on) 
        
    elif image_type == InputFeature.TOP2:
        f_img = RFilter.tophat(img, nml_on=nml_on) 
          
    elif image_type == InputFeature.TOP3:
        f_img = RFilter.median_tophat(img, 4, 4, nml_on=nml_on) 

    elif image_type == InputFeature.TOP4:
        f_img = RFilter.median_tophat(img, 4, 2, nml_on=nml_on) 

    elif image_type == InputFeature.SBLX:
        f_img = RFilter.sobel(img, FilterKernel.SBLX33, nml_on=nml_on) 
    
    elif image_type == InputFeature.SBLY:
        f_img = RFilter.sobel(img, FilterKernel.SBLY33, nml_on=nml_on) 

    elif image_type == InputFeature.SBLM:
        f_img = RFilter.sobel_mag(img, nml_on=nml_on) 
   
    elif image_type == InputFeature.SBLD:
        f_img = RFilter.sobel_dir(img, nml_on=nml_on) 

    elif image_type == InputFeature.SBL1:
        f_img = RFilter.sobel(img, FilterKernel.SBLX35, nml_on=nml_on) 

    elif image_type == InputFeature.SBL2:
        f_img = RFilter.sobel(img, FilterKernel.SBLY53, nml_on=nml_on) 
    
    elif image_type == InputFeature.SBL3:
        f_img = RFilter.sobel(img, FilterKernel.SBLX55, nml_on=nml_on) 
    
    elif image_type == InputFeature.SBL4:
        f_img = RFilter.sobel(img, FilterKernel.SBLY55, nml_on=nml_on) 

    elif image_type == InputFeature.LPL1:
        f_img = RFilter.laplacian(img, FilterKernel.LPL4, nml_on=nml_on) 

    elif image_type == InputFeature.LPL2:
        f_img = RFilter.laplacian(img, FilterKernel.LPL8, nml_on=nml_on) 

    elif image_type == InputFeature.MEA1:
        f_img = RFilter.mean(img, (3,3), nml_on=nml_on) 

    elif image_type == InputFeature.MEA2:
        f_img = RFilter.mean(img, (5,5), nml_on=nml_on) 

    elif image_type == InputFeature.GAU1:
        f_img = RFilter.gaussian(img, 3, nml_on=nml_on) 

    elif image_type == InputFeature.GAU2:
        f_img = RFilter.gaussian(img, 5, nml_on=nml_on) 
    
    elif image_type == InputFeature.MED1:
        f_img = RFilter.median(img, ksize=1, nml_on=nml_on) 

    elif image_type == InputFeature.MED2:
        f_img = RFilter.median(img, ksize=2, nml_on=nml_on) 

    elif image_type == InputFeature.LBP1:
        f_img = RFilter.median_lbp(img, nml_on=nml_on) 

    elif image_type == InputFeature.LBP2:
        f_img = RFilter.local_binary_pattern(img, 1, nml_on=nml_on) 

    elif image_type == InputFeature.LBP3:
        f_img = RFilter.local_binary_pattern(img, 2, nml_on=nml_on) 

    elif image_type == InputFeature.ETC1:
        f_img = RFilter.exponential_tone_curve(img, 2.0, nml_on=nml_on) 

    elif image_type == InputFeature.ETC2:
        f_img = RFilter.exponential_tone_curve(img, 0.5, nml_on=nml_on) 

    elif image_type == InputFeature.STC1:
        f_img = RFilter.sigmoid_tone_curve(img, 0.0, 2.0, nml_on=nml_on) 

    elif image_type == InputFeature.STC2:
        f_img = RFilter.sigmoid_tone_curve(img, 1.0, 2.0, nml_on=nml_on) 

    elif image_type == InputFeature.HGF_:
        f_img = RFilter.hist_flat(img, nml_on=nml_on) 
            
    elif image_type == InputFeature.NGP_:
        f_img = RFilter.negaposi(img, nml_on=nml_on) 

    elif image_type == InputFeature.POS1:
        f_img = RFilter.posterization(img, 10, nml_on=nml_on) 

    elif image_type == InputFeature.POS2:
        f_img = RFilter.posterization(img, 20, nml_on=nml_on) 

    elif image_type == InputFeature.POS3:
        f_img = RFilter.posterization(img, 30, nml_on=nml_on) 

    elif image_type == InputFeature.SOL_:
        f_img = RFilter.solarization(img, nml_on=nml_on) 

    elif image_type == InputFeature.EMB1:
        f_img = RFilter.emboss(img, FilterKernel.EMB35, nml_on=nml_on) 

    elif image_type == InputFeature.EMB2:
        f_img = RFilter.emboss(img, FilterKernel.EMB53, nml_on=nml_on) 

    elif image_type == InputFeature.EMB3:
        f_img = RFilter.emboss(img, FilterKernel.EMB33, nml_on=nml_on) 

    elif image_type == InputFeature.KNN1:
        f_img = RFilter.knn_ave(img, 3, 3, nml_on=nml_on) 

    elif image_type == InputFeature.KNN2:
        f_img = RFilter.knn_ave(img, 5, 5, nml_on=nml_on) 

    elif image_type == InputFeature.BLT1:
        f_img = RFilter.bilateral(img, 3, 1.0, 2.0, nml_on=nml_on) 

    elif image_type == InputFeature.BLT2:
        f_img = RFilter.bilateral(img, 5, 1.0, 2.0, nml_on=nml_on) 

    elif image_type == InputFeature.OOO_:
        f_img = RFilter.black(img) 

    ################# Custom added filters

    # CANNY FILTER
    elif image_type == InputFeature.CAN1:
        f_img = RFilter.canny(img, 50, 150)

    elif image_type == InputFeature.CAN2:
        f_img = RFilter.canny(img, 100, 200)

    elif image_type == InputFeature.CAN3:
        f_img = RFilter.canny(img, 150, 250)

    #PREWITT FILTER
    elif image_type == InputFeature.PRE1:
        f_img = RFilter.prewitt(img, 0)

    elif image_type == InputFeature.PRE2:
        f_img = RFilter.prewitt(img, 1)

    elif image_type == InputFeature.PRE3:
        f_img = RFilter.prewitt(img, 2)

    elif image_type == InputFeature.PRE4:
        f_img = RFilter.prewitt(img, 3)

    #UNSHARPENING FILTER
    elif image_type == InputFeature.UNS1:
        f_img = RFilter.unsharp_masking(img, (3,3), 1.0, 1.5, -0.5)
    elif image_type == InputFeature.UNS2:
        f_img = RFilter.unsharp_masking(img, (7,7), 1.0, 1.5, -0.5)
    elif image_type == InputFeature.UNS3:
        f_img = RFilter.unsharp_masking(img, (5,5), 2.0, 1.5, -0.5)
    elif image_type == InputFeature.UNS4:
        f_img = RFilter.unsharp_masking(img, (5,5), 1.0, 1.3, -0.3)
    elif image_type == InputFeature.UNS5:
        f_img = RFilter.unsharp_masking(img, (5,5), 1.0, 2.0, -1.0)

    
    #EROSION FILTER
    elif image_type == InputFeature.ERO1:
        f_img = RFilter.erosion(img, 0)
    elif image_type == InputFeature.ERO2:
        f_img = RFilter.erosion(img, 1)
    elif image_type == InputFeature.ERO3:
        f_img = RFilter.erosion(img, 2)
    elif image_type == InputFeature.ERO4:
        f_img = RFilter.erosion(img, 3)
    elif image_type == InputFeature.ERO5:
        f_img = RFilter.erosion(img, 4)
    elif image_type == InputFeature.ERO6:
        f_img = RFilter.erosion(img, 5)
    
    #OPENING FILTER
    elif image_type == InputFeature.OPN1:
        f_img = RFilter.opening(img, 0)
    elif image_type == InputFeature.OPN2:
        f_img = RFilter.opening(img, 1)
    elif image_type == InputFeature.OPN3:
        f_img = RFilter.opening(img, 2)
    elif image_type == InputFeature.OPN4:
        f_img = RFilter.opening(img, 3)
    elif image_type == InputFeature.OPN5:
        f_img = RFilter.opening(img, 4)
    
    #CLOSING FILTER
    elif image_type == InputFeature.CLO1:
        f_img = RFilter.closing(img, 0)
    elif image_type == InputFeature.CLO2:
        f_img = RFilter.closing(img, 1)
    elif image_type == InputFeature.CLO3:
        f_img = RFilter.closing(img, 2)
    elif image_type == InputFeature.CLO4:
        f_img = RFilter.closing(img, 3)
    elif image_type == InputFeature.CLO5:
        f_img = RFilter.closing(img, 4)
        
    #FOURIER FILTER
    elif image_type == InputFeature.FOU1:
        f_img= RFilter.fourier(img, filter_type='low_pass', d0=50)
    elif image_type == InputFeature.FOU2:
        f_img = RFilter.fourier(img, filter_type='high_pass', d0=50)
    elif image_type == InputFeature.FOU3:
        f_img= RFilter.fourier(img, filter_type='band_pass', params=(50, 100))
    elif image_type == InputFeature.FOU4:
        f_img = RFilter.fourier(img, filter_type='notch', params=((100, 100), 20))

    #BOXING FILTER
    # elif image_type == InputFeature.BOX1:
    #     f_img = RFilter.box(img, (3,3))
    # elif image_type == InputFeature.BOX2:
    #     f_img = RFilter.box(img, (5,5))
    # elif image_type == InputFeature.BOX3:
    #     f_img = RFilter.box(img, (7,7))
    # elif image_type == InputFeature.BOX4:
    #     f_img = RFilter.box(img, (9,9))

    #SCHARR GRADIENT FILTER
    elif image_type == InputFeature.SCH1:
        f_img = RFilter.scharr_grad(img, 1)
    elif image_type == InputFeature.SCH2:
        f_img = RFilter.scharr_grad(img, 10)
    elif image_type == InputFeature.SCH3:
        f_img = RFilter.scharr_grad(img, 0.1)
    elif image_type == InputFeature.SCH4:
        f_img = RFilter.scharr_grad(img, 5)

    #ROBERTS CROSS FILTER
    elif image_type == InputFeature.ROB1:
        f_img = RFilter.roberts_cross(img, 1)
    elif image_type == InputFeature.ROB2:
        f_img = RFilter.roberts_cross(img, 2)
    elif image_type == InputFeature.ROB3:
        f_img = RFilter.roberts_cross(img, 3)
    elif image_type == InputFeature.ROB4:
        f_img = RFilter.roberts_cross(img, 4)

    #MIN FILTER
    elif image_type == InputFeature.MIN1:
        f_img = RFilter.min(img, (3,3))
    elif image_type == InputFeature.MIN2:
        f_img = RFilter.min(img, (5,5))
    elif image_type == InputFeature.MIN3:
        f_img = RFilter.min(img, (7,7))
    elif image_type == InputFeature.MIN4:
        f_img = RFilter.min(img, (9,9))
    
    #MAX FILTER
    elif image_type == InputFeature.MAX1:
        f_img = RFilter.max(img, (3,3))
    elif image_type == InputFeature.MAX2:
        f_img = RFilter.max(img, (5,5))
    elif image_type == InputFeature.MAX3:
        f_img = RFilter.max(img, (7,7))
    elif image_type == InputFeature.MAX4:
        f_img = RFilter.max(img, (9,9))

    # MORPH GRADIENT FILTER
    elif image_type == InputFeature.MRG1:
        f_img = RFilter.morph_grad(img, (3,3))
    elif image_type == InputFeature.MRG2:
        f_img = RFilter.morph_grad(img, (5,5))
    elif image_type == InputFeature.MRG3:
        f_img = RFilter.morph_grad(img, (7,7))
    elif image_type == InputFeature.MRG4:
        f_img = RFilter.morph_grad(img, (9,9))

    # MORPH LAPLACIAN FILTER
    elif image_type == InputFeature.MRL1:
        f_img = RFilter.morph_laplacian(img, (3,3))
    elif image_type == InputFeature.MRL2:
        f_img = RFilter.morph_laplacian(img, (5,5))
    elif image_type == InputFeature.MRL3:
        f_img = RFilter.morph_laplacian(img, (7,7))
    elif image_type == InputFeature.MRL4:
        f_img = RFilter.morph_laplacian(img, (9,9))

    # BOTTOM HAT FILTER
    elif image_type == InputFeature.BTM1:
        f_img = RFilter.bottom_hat_transform(img, (3,3))
    elif image_type == InputFeature.BTM2:
        f_img = RFilter.bottom_hat_transform(img, (5,5))
    elif image_type == InputFeature.BTM3:
        f_img = RFilter.bottom_hat_transform(img, (7,7))
    elif image_type == InputFeature.BTM4:
        f_img = RFilter.bottom_hat_transform(img, (9,9))

    #DISTANCE TRANSFORM
    elif image_type == InputFeature.DST_:
        f_img = RFilter.distance_transform(img)
    
    #HOMOMORPHIC FILTER
    elif image_type == InputFeature.HOM_:
        f_img = RFilter.homomorphic(img)
    # STRUCTURE TENSOR FILTER
    #elif image_type == InputFeature.STR_:
    #   f_img = RFilter.structure_tensor(img)
    # RICHARDSON LUCY FILTER
    elif image_type == InputFeature.RIC_:
        f_img = RFilter.richardson_lucy(img)

    else:
        print("image_type is wrong!")
        return
    
    # print(f_img.shape)
    return f_img
    
def list_images(image_list, image_type, nml_on=True):
    # 画像処理を施す
    if image_type == InputFeature.GRY_:
        f_img_list = [RFilter.gray(img) for img in image_list]

    elif image_type == InputFeature.NML1:
        f_img_list = [RFilter.normalization(img, 1) for img in image_list]

    elif image_type == InputFeature.NML2:
        f_img_list = [RFilter.normalization(img, 2) for img in image_list]

    elif image_type == InputFeature.NML3:
        f_img_list = [RFilter.normalization(img, 3) for img in image_list]

    elif image_type == InputFeature.TOP1:
        f_img_list = [RFilter.median_tophat(img, 1, 4, nml_on=nml_on) for img in image_list]

    elif image_type == InputFeature.TOP2:
        f_img_list = [RFilter.tophat(img, nml_on=nml_on) for img in image_list]
 
    elif image_type == InputFeature.TOP3:
        f_img_list = [RFilter.median_tophat(img, 4, 4, nml_on=nml_on) for img in image_list]

    elif image_type == InputFeature.TOP4:
        f_img_list = [RFilter.median_tophat(img, 4, 2, nml_on=nml_on) for img in image_list]

    elif image_type == InputFeature.SBLX:
        f_img_list = [RFilter.sobel(img, FilterKernel.SBLX33, nml_on=nml_on) for img in image_list]

    elif image_type == InputFeature.SBLY:
        f_img_list = [RFilter.sobel(img, FilterKernel.SBLY33, nml_on=nml_on) for img in image_list]

    elif image_type == InputFeature.SBLM:
        f_img_list = [RFilter.sobel_mag(img, nml_on=nml_on) for img in image_list]

    elif image_type == InputFeature.SBLD:
        f_img_list = [RFilter.sobel_dir(img, nml_on=nml_on) for img in image_list]

    elif image_type == InputFeature.SBL1:
        f_img_list = [RFilter.sobel(img, FilterKernel.SBLX35, nml_on=nml_on) for img in image_list]

    elif image_type == InputFeature.SBL2:
        f_img_list = [RFilter.sobel(img, FilterKernel.SBLY53, nml_on=nml_on) for img in image_list]

    elif image_type == InputFeature.SBL3:
        f_img_list = [RFilter.sobel(img, FilterKernel.SBLX55, nml_on=nml_on) for img in image_list]
    
    elif image_type == InputFeature.SBL4:
        f_img_list = [RFilter.sobel(img, FilterKernel.SBLY55, nml_on=nml_on) for img in image_list]

    elif image_type == InputFeature.LPL1:
        f_img_list = [RFilter.laplacian(img, FilterKernel.LPL4, nml_on=nml_on) for img in image_list]

    elif image_type == InputFeature.LPL2:
        f_img_list = [RFilter.laplacian(img, FilterKernel.LPL8, nml_on=nml_on) for img in image_list]

    elif image_type == InputFeature.MEA1:
        f_img_list = [RFilter.mean(img, (3,3), nml_on=nml_on) for img in image_list]

    elif image_type == InputFeature.MEA2:
        f_img_list = [RFilter.mean(img, (5,5), nml_on=nml_on) for img in image_list]

    elif image_type == InputFeature.GAU1:
        f_img_list = [RFilter.gaussian(img, 3, nml_on=nml_on) for img in image_list]

    elif image_type == InputFeature.GAU2:
        f_img_list = [RFilter.gaussian(img, 5, nml_on=nml_on) for img in image_list]

    elif image_type == InputFeature.MED1:
        f_img_list = [RFilter.median(img, ksize=1, nml_on=nml_on) for img in image_list]

    elif image_type == InputFeature.MED2:
        f_img_list = [RFilter.median(img, ksize=2, nml_on=nml_on) for img in image_list]

    elif image_type == InputFeature.LBP1:
        f_img_list = [RFilter.median_lbp(img, nml_on=nml_on) for img in image_list]

    elif image_type == InputFeature.LBP2:
        f_img_list = [RFilter.local_binary_pattern(img, 1, nml_on=nml_on) for img in image_list]

    elif image_type == InputFeature.LBP3:
        f_img_list = [RFilter.local_binary_pattern(img, 2, nml_on=nml_on) for img in image_list]

    elif image_type == InputFeature.ETC1:
        f_img_list = [RFilter.exponential_tone_curve(img, 2.0, nml_on=nml_on) for img in image_list]

    elif image_type == InputFeature.ETC2:
        f_img_list = [RFilter.exponential_tone_curve(img, 0.5, nml_on=nml_on) for img in image_list]

    elif image_type == InputFeature.STC1:
        f_img_list = [RFilter.sigmoid_tone_curve(img, 0.0, 2.0, nml_on=nml_on) for img in image_list]

    elif image_type == InputFeature.STC2:
        f_img_list = [RFilter.sigmoid_tone_curve(img, 1.0, 2.0, nml_on=nml_on) for img in image_list]

    elif image_type == InputFeature.HGF_:
        f_img_list = [RFilter.hist_flat(img, nml_on=nml_on) for img in image_list]
            
    elif image_type == InputFeature.NGP_:
        f_img_list = [RFilter.negaposi(img, nml_on=nml_on) for img in image_list]

    elif image_type == InputFeature.POS1:
        f_img_list = [RFilter.posterization(img, 10, nml_on=nml_on) for img in image_list]

    elif image_type == InputFeature.POS2:
        f_img_list = [RFilter.posterization(img, 20, nml_on=nml_on) for img in image_list]

    elif image_type == InputFeature.POS3:
        f_img_list = [RFilter.posterization(img, 30, nml_on=nml_on) for img in image_list]

    elif image_type == InputFeature.SOL_:
        f_img_list = [RFilter.solarization(img, nml_on=nml_on) for img in image_list]

    elif image_type == InputFeature.EMB1:
        f_img_list = [RFilter.emboss(img, FilterKernel.EMB35, nml_on=nml_on) for img in image_list]

    elif image_type == InputFeature.EMB2:
        f_img_list = [RFilter.emboss(img, FilterKernel.EMB53, nml_on=nml_on) for img in image_list]

    elif image_type == InputFeature.EMB3:
        f_img_list = [RFilter.emboss(img, FilterKernel.EMB33, nml_on=nml_on) for img in image_list]
 
    elif image_type == InputFeature.KNN1:
        f_img_list = [RFilter.knn_ave(img, 3, 3, nml_on=nml_on) for img in image_list]
    
    elif image_type == InputFeature.KNN2:
        f_img_list = [RFilter.knn_ave(img, 5, 5, nml_on=nml_on) for img in image_list]

    elif image_type == InputFeature.BLT1:
        f_img_list = [RFilter.bilateral(img, 3, 1.0, 2.0, nml_on=nml_on) for img in image_list]

    elif image_type == InputFeature.BLT2:
        f_img_list = [RFilter.bilateral(img, 5, 1.0, 2.0, nml_on=nml_on) for img in image_list]

    elif image_type == InputFeature.OOO_:
        f_img_list = [RFilter.black(img) for img in image_list]

    else:
        print("image_type is wrong!")
        return
    
    # f_img_list = np.asarray(f_img_list, dtype=np.float64)
    # print(f_img_list.shape)
    return f_img_list

class PreProcImage:
    """
    特徴画像のデータセットを作成する．
    [注意] このクラスは今使っていない．色々バグっているので使わないほうがいい．
    ## <使い方例>
    ```
    preprocimage = PreProcImage()   # PreProcImageのコンストラクタの作成
    preprocimage.set_image(image_root_dir, train_filenames, test_filenames)   # 入力データのパス設定
    image_train, image_test = preprocimage.filtering(inputfeaure, featureDir, nml_on=True)    # フィルタリングを実行
    ```
    """
    def __init__(self) -> None:
        pass

    def set_image(self, input_directory, train_filenames, test_filenames, val_filenames=None):
        """
        対象の画像のセット．学習データ，テストデータ，(検証データ)
        
        :param input_directory: 画像が入っているフォルダのパス
        :param train_filenames: トレーニング画像のファイル名が記述されたcsvファイルのパス
        :param test_filenames: テスト画像のファイル名が記述されたcsvファイルのパス
        :param val_filenames: 検証画像のファイル名が記述されたcsvファイルのパス

        """
        input_image_dir = input_directory + 'image/'  # 入力画像が入っているディレクトリパス
        input_label_dir = input_directory + 'label/'  # 入力画像のラベルが入っているディレクトリパス
        self.input_label_dir = input_label_dir

        image_train, image_train_filenames = read_csv_image(input_image_dir, train_filenames)
        label_train, label_train_filenames = read_csv_image(input_label_dir, train_filenames)

        train_groupfilenames = train_filenames.split('/')
        train_groupfilename  = train_groupfilenames[len(train_groupfilenames)-1]
        train_groupname      = train_groupfilename.split('.')[0]

        self.image_train = image_train
        self.label_train = label_train
        
        self.train_groupname = train_groupname
        self.train_casenames = label_train_filenames

        image_test, image_test_filenames = read_csv_image(input_image_dir, test_filenames)
        label_test, label_test_filenames = read_csv_image(input_label_dir, test_filenames)

        test_groupfilenames = test_filenames.split('/')
        test_groupfilename  = test_groupfilenames[len(test_groupfilenames)-1]
        test_groupname      = test_groupfilename.split('.')[0]

        self.image_test = image_test
        self.label_test = label_test

        self.test_groupname = test_groupname
        self.test_casenames = label_test_filenames


        if val_filenames is not None:
            image_val, image_val_filenames = read_csv_image(input_image_dir, val_filenames)
            label_val, label_val_filenames = read_csv_image(input_label_dir, val_filenames)

            val_groupfilenames = val_filenames.split('/')
            val_groupfilename  = val_groupfilenames[len(val_groupfilenames)-1]
            val_groupname      = val_groupfilename.split('.')[0]

            self.image_val = image_val
            self.label_val = label_val

            self.val_groupname = val_groupname
            self.val_casenames = label_val_filenames
        else :
            self.image_val = None


    def filtering(self, image_type, output_dir, nml_on=True):
        """
        フィルタリングをする．このときに生成された画像はそれぞれoutput_dirのなかにcsvファイルのファイル名
        のフォルダにそれぞれ画像が保存される．
        
        :param image_type: フィルタ．InputFeature内のフィルタを選択．
        :param output_dir: フィルタリングした画像の保存先
        :param nml_on: 正規化をするか

        :returns: image_train: numpuy形式のトレーニングデータ(フィルタリング済)．
        :returns: image_test: numpuy形式のテストデータ(フィルタリング済)．
        :returns: (image_val: numpuy形式のバリデーションデータ(フィルタリング済)．set_imageでval_filenames=Nonenに
            していればこれは返されない．)
        """
        image_train = self.image_train
        label_train = self.label_train
        image_test = self.image_test
        label_test = self.label_test

        if self.image_val is not None:
            image_val = self.image_val
            label_val = self.label_val

        filtering_train_file = output_dir + image_type.value + '/' + self.train_groupname
        filtering_test_file  = output_dir + image_type.value + '/' + self.test_groupname
        if os.path.exists(filtering_train_file) and os.path.exists(filtering_test_file):
            print("The file(" + image_type.value + ") already exists!")
            self.image_train = ReadFile.directory_images(filtering_train_file, IMAGE_SIZE, 'Gray')
            self.image_test = ReadFile.directory_images(filtering_test_file, IMAGE_SIZE, 'Gray')
            # バグ取れたらコメントアウト外す
            # return
        
        if self.image_val is not None:
            filtering_val_file  = output_dir + image_type.value + '/' + self.val_groupname
            if os.path.exists(filtering_train_file) and os.path.exists(filtering_test_file):
                self.image_val = ReadFile.directory_images(filtering_val_file, IMAGE_SIZE, 'Gray')

        # 画像処理を施す
        if image_type == InputFeature.GRY_:
            image_train = [RFilter.gray(img) for img in image_train]
            image_test  = [RFilter.gray(img) for img in image_test]
            if self.image_val is not None:
                image_val  = [RFilter.gray(img) for img in image_val]

        elif image_type == InputFeature.NML1:
            image_train = [RFilter.normalization(img, 1) for img in image_train]
            image_test  = [RFilter.normalization(img, 1) for img in image_test]
            if self.image_val is not None:
                image_val  = [RFilter.normalization(img, 1) for img in image_val]

        elif image_type == InputFeature.NML2:
            image_train = [RFilter.normalization(img, 2) for img in image_train]
            image_test  = [RFilter.normalization(img, 2) for img in image_test]
            if self.image_val is not None:
                image_val  = [RFilter.normalization(img, 2) for img in image_val]

        elif image_type == InputFeature.NML3:
            image_train = [RFilter.normalization(img, 3) for img in image_train]
            image_test  = [RFilter.normalization(img, 3) for img in image_test]
            if self.image_val is not None:
                image_val  = [RFilter.normalization(img, 3) for img in image_val]

        elif image_type == InputFeature.TOP1:
            image_train = [RFilter.median_tophat(img, 1, 4, nml_on=nml_on) for img in image_train]
            image_test  = [RFilter.median_tophat(img, 1, 4, nml_on=nml_on) for img in image_test]
            if self.image_val is not None:
                image_val  = [RFilter.median_tophat(img, 1, 4, nml_on=nml_on) for img in image_val]

        elif image_type == InputFeature.TOP2:
            image_train = [RFilter.tophat(img, nml_on=nml_on) for img in image_train]
            image_test  = [RFilter.tophat(img, nml_on=nml_on) for img in image_test]    
            if self.image_val is not None:
                image_val  = [RFilter.tophat(img, nml_on=nml_on) for img in image_val]    

        elif image_type == InputFeature.TOP3:
            image_train = [RFilter.median_tophat(img, 4, 4, nml_on=nml_on) for img in image_train]
            image_test  = [RFilter.median_tophat(img, 4, 4, nml_on=nml_on) for img in image_test]
            if self.image_val is not None:
                image_val  = [RFilter.median_tophat(img, 4, 4, nml_on=nml_on) for img in image_val]

        elif image_type == InputFeature.TOP4:
            image_train = [RFilter.median_tophat(img, 4, 2, nml_on=nml_on) for img in image_train]
            image_test  = [RFilter.median_tophat(img, 4, 2, nml_on=nml_on) for img in image_test]
            if self.image_val is not None:
                image_val  = [RFilter.median_tophat(img, 4, 2, nml_on=nml_on) for img in image_val]

        elif image_type == InputFeature.SBLX:
            image_train = [RFilter.sobel(img, FilterKernel.SBLX33, nml_on=nml_on) for img in image_train]
            image_test  = [RFilter.sobel(img, FilterKernel.SBLX33, nml_on=nml_on)for img in image_test]
            if self.image_val is not None:
                image_val  = [RFilter.sobel(img, FilterKernel.SBLX33, nml_on=nml_on)for img in image_val]
        
        elif image_type == InputFeature.SBLY:
            image_train = [RFilter.sobel(img, FilterKernel.SBLY33, nml_on=nml_on) for img in image_train]
            image_test  = [RFilter.sobel(img, FilterKernel.SBLY33, nml_on=nml_on)for img in image_test]
            if self.image_val is not None:
                image_val  = [RFilter.sobel(img, FilterKernel.SBLY33, nml_on=nml_on)for img in image_val]

        elif image_type == InputFeature.SBLM:
            image_train = [RFilter.sobel_mag(img, nml_on=nml_on) for img in image_train]
            image_test  = [RFilter.sobel_mag(img, nml_on=nml_on)for img in image_test]
            if self.image_val is not None:
                image_val  = [RFilter.sobel_mag(img, nml_on=nml_on)for img in image_val]

        elif image_type == InputFeature.SBLD:
            image_train = [RFilter.sobel_dir(img, nml_on=nml_on) for img in image_train]
            image_test  = [RFilter.sobel_dir(img, nml_on=nml_on)for img in image_test]
            if self.image_val is not None:
                image_val  = [RFilter.sobel_dir(img, nml_on=nml_on)for img in image_val]

        elif image_type == InputFeature.SBL1:
            image_train = [RFilter.sobel(img, FilterKernel.SBLX35, nml_on=nml_on) for img in image_train]
            image_test  = [RFilter.sobel(img, FilterKernel.SBLX35, nml_on=nml_on)for img in image_test]
            if self.image_val is not None:
                image_val  = [RFilter.sobel(img, FilterKernel.SBLX35, nml_on=nml_on)for img in image_val]

        elif image_type == InputFeature.SBL2:
            image_train = [RFilter.sobel(img, FilterKernel.SBLY53, nml_on=nml_on) for img in image_train]
            image_test  = [RFilter.sobel(img, FilterKernel.SBLY53, nml_on=nml_on)for img in image_test]
            if self.image_val is not None:
                image_val  = [RFilter.sobel(img, FilterKernel.SBLY53, nml_on=nml_on)for img in image_val]
        
        elif image_type == InputFeature.SBL3:
            image_train = [RFilter.sobel(img, FilterKernel.SBLX55, nml_on=nml_on) for img in image_train]
            image_test  = [RFilter.sobel(img, FilterKernel.SBLX55, nml_on=nml_on)for img in image_test]
            if self.image_val is not None:
                image_val  = [RFilter.sobel(img, FilterKernel.SBLX55, nml_on=nml_on)for img in image_val]
        
        elif image_type == InputFeature.SBL4:
            image_train = [RFilter.sobel(img, FilterKernel.SBLY55, nml_on=nml_on) for img in image_train]
            image_test  = [RFilter.sobel(img, FilterKernel.SBLY55, nml_on=nml_on)for img in image_test]
            if self.image_val is not None:
                image_val  = [RFilter.sobel(img, FilterKernel.SBLY55, nml_on=nml_on)for img in image_val]

        elif image_type == InputFeature.LPL1:
            image_train = [RFilter.laplacian(img, FilterKernel.LPL4, nml_on=nml_on) for img in image_train]
            image_test  = [RFilter.laplacian(img, FilterKernel.LPL4, nml_on=nml_on) for img in image_test]
            if self.image_val is not None:
                image_val  = [RFilter.laplacian(img, FilterKernel.LPL4, nml_on=nml_on) for img in image_val]

        elif image_type == InputFeature.LPL2:
            image_train = [RFilter.laplacian(img, FilterKernel.LPL8, nml_on=nml_on) for img in image_train]
            image_test  = [RFilter.laplacian(img, FilterKernel.LPL8, nml_on=nml_on) for img in image_test]
            if self.image_val is not None:
                image_val = [RFilter.laplacian(img, FilterKernel.LPL8, nml_on=nml_on) for img in image_val]

        elif image_type == InputFeature.MEA1:
            image_train = [RFilter.mean(img, (3,3), nml_on=nml_on) for img in image_train]
            image_test  = [RFilter.mean(img, (3,3), nml_on=nml_on) for img in image_test]
            if self.image_val is not None:
                image_val  = [RFilter.mean(img, (3,3), nml_on=nml_on) for img in image_val]

        elif image_type == InputFeature.MEA2:
            image_train = [RFilter.mean(img, (5,5), nml_on=nml_on) for img in image_train]
            image_test  = [RFilter.mean(img, (5,5), nml_on=nml_on) for img in image_test]
            if self.image_val is not None:
                image_val  = [RFilter.mean(img, (5,5), nml_on=nml_on) for img in image_val]

        elif image_type == InputFeature.GAU1:
            image_train = [RFilter.gaussian(img, 3, nml_on=nml_on) for img in image_train]
            image_test  = [RFilter.gaussian(img, 3, nml_on=nml_on) for img in image_test]  
            if self.image_val is not None:
                image_val  = [RFilter.gaussian(img, 3, nml_on=nml_on) for img in image_val]  

        elif image_type == InputFeature.GAU2:
            image_train = [RFilter.gaussian(img, 5, nml_on=nml_on) for img in image_train]
            image_test  = [RFilter.gaussian(img, 5, nml_on=nml_on) for img in image_test] 
            if self.image_val is not None:
                image_val  = [RFilter.gaussian(img, 5, nml_on=nml_on) for img in image_val] 
        
        elif image_type == InputFeature.MED1:
            image_train = [RFilter.median(img, ksize=1, nml_on=nml_on) for img in image_train]
            image_test  = [RFilter.median(img, ksize=1, nml_on=nml_on) for img in image_test] 
            if self.image_val is not None:
                image_val  = [RFilter.median(img, ksize=1, nml_on=nml_on) for img in image_val] 

        elif image_type == InputFeature.MED2:
            image_train = [RFilter.median(img, ksize=2, nml_on=nml_on) for img in image_train]
            image_test  = [RFilter.median(img, ksize=2, nml_on=nml_on) for img in image_test]
            if self.image_val is not None:
                image_val  = [RFilter.median(img, ksize=2, nml_on=nml_on) for img in image_val]

        elif image_type == InputFeature.LBP1:
            image_train = [RFilter.median_lbp(img, nml_on=nml_on) for img in image_train]
            image_test  = [RFilter.median_lbp(img, nml_on=nml_on) for img in image_test] 
            if self.image_val is not None:
                image_val  = [RFilter.median_lbp(img, nml_on=nml_on) for img in image_val] 

        elif image_type == InputFeature.LBP2:
            image_train = [RFilter.local_binary_pattern(img, 1, nml_on=nml_on) for img in image_train]
            image_test  = [RFilter.local_binary_pattern(img, 1, nml_on=nml_on) for img in image_test] 
            if self.image_val is not None:
                image_val  = [RFilter.local_binary_pattern(img, 1, nml_on=nml_on) for img in image_val] 

        elif image_type == InputFeature.LBP3:
            image_train = [RFilter.local_binary_pattern(img, 2, nml_on=nml_on) for img in image_train]
            image_test  = [RFilter.local_binary_pattern(img, 2, nml_on=nml_on) for img in image_test] 
            if self.image_val is not None:
                image_val  = [RFilter.local_binary_pattern(img, 2, nml_on=nml_on) for img in image_val] 

        elif image_type == InputFeature.ETC1:
            image_train = [RFilter.exponential_tone_curve(img, 2.0, nml_on=nml_on) for img in image_train]
            image_test  = [RFilter.exponential_tone_curve(img, 2.0, nml_on=nml_on) for img in image_test] 
            if self.image_val is not None:
                image_val  = [RFilter.exponential_tone_curve(img, 2.0, nml_on=nml_on) for img in image_val] 

        elif image_type == InputFeature.ETC2:
            image_train = [RFilter.exponential_tone_curve(img, 0.5, nml_on=nml_on) for img in image_train]
            image_test  = [RFilter.exponential_tone_curve(img, 0.5, nml_on=nml_on) for img in image_test] 
            if self.image_val is not None:
                image_val  = [RFilter.exponential_tone_curve(img, 0.5, nml_on=nml_on) for img in image_val] 

        elif image_type == InputFeature.STC1:
            image_train = [RFilter.sigmoid_tone_curve(img, 0.0, 2.0, nml_on=nml_on) for img in image_train]
            image_test  = [RFilter.sigmoid_tone_curve(img, 0.0, 2.0, nml_on=nml_on) for img in image_test] 
            if self.image_val is not None:
                image_val  = [RFilter.sigmoid_tone_curve(img, 0.0, 2.0, nml_on=nml_on) for img in image_val] 

        elif image_type == InputFeature.STC2:
            image_train = [RFilter.sigmoid_tone_curve(img, 1.0, 2.0, nml_on=nml_on) for img in image_train]
            image_test  = [RFilter.sigmoid_tone_curve(img, 1.0, 2.0, nml_on=nml_on) for img in image_test] 
            if self.image_val is not None:
                image_val  = [RFilter.sigmoid_tone_curve(img, 1.0, 2.0, nml_on=nml_on) for img in image_val] 

        elif image_type == InputFeature.HGF_:
            image_train = [RFilter.hist_flat(img, nml_on=nml_on) for img in image_train]
            image_test  = [RFilter.hist_flat(img, nml_on=nml_on) for img in image_test] 
            if self.image_val is not None:
                image_val  = [RFilter.hist_flat(img, nml_on=nml_on) for img in image_val] 
                
        elif image_type == InputFeature.NGP_:
            image_train = [RFilter.negaposi(img, nml_on=nml_on) for img in image_train]
            image_test  = [RFilter.negaposi(img, nml_on=nml_on) for img in image_test] 
            if self.image_val is not None:
                image_val  = [RFilter.negaposi(img, nml_on=nml_on) for img in image_val] 

        elif image_type == InputFeature.POS1:
            image_train = [RFilter.posterization(img, 10, nml_on=nml_on) for img in image_train]
            image_test  = [RFilter.posterization(img, 10, nml_on=nml_on)  for img in image_test] 
            if self.image_val is not None:
                image_val  = [RFilter.posterization(img, 10, nml_on=nml_on)  for img in image_val]

        elif image_type == InputFeature.POS2:
            image_train = [RFilter.posterization(img, 20, nml_on=nml_on) for img in image_train]
            image_test  = [RFilter.posterization(img, 20, nml_on=nml_on)  for img in image_test] 
            if self.image_val is not None:
                image_val  = [RFilter.posterization(img, 20, nml_on=nml_on)  for img in image_val] 

        elif image_type == InputFeature.POS3:
            image_train = [RFilter.posterization(img, 30, nml_on=nml_on) for img in image_train]
            image_test  = [RFilter.posterization(img, 30, nml_on=nml_on)  for img in image_test] 
            if self.image_val is not None:
                image_val  = [RFilter.posterization(img, 30, nml_on=nml_on)  for img in image_val] 

        elif image_type == InputFeature.SOL_:
            image_train = [RFilter.solarization(img, nml_on=nml_on) for img in image_train]
            image_test  = [RFilter.solarization(img, nml_on=nml_on)  for img in image_test] 
            if self.image_val is not None:
                image_val  = [RFilter.solarization(img, nml_on=nml_on)  for img in image_val] 

        elif image_type == InputFeature.EMB1:
            image_train = [RFilter.emboss(img, FilterKernel.EMB35, nml_on=nml_on) for img in image_train]
            image_test  = [RFilter.emboss(img, FilterKernel.EMB35, nml_on=nml_on)  for img in image_test] 
            if self.image_val is not None:
                image_val  = [RFilter.emboss(img, FilterKernel.EMB35, nml_on=nml_on)  for img in image_val] 
    
        elif image_type == InputFeature.EMB2:
            image_train = [RFilter.emboss(img, FilterKernel.EMB53, nml_on=nml_on) for img in image_train]
            image_test  = [RFilter.emboss(img, FilterKernel.EMB53, nml_on=nml_on)  for img in image_test] 
            if self.image_val is not None:
                image_val  = [RFilter.emboss(img, FilterKernel.EMB53, nml_on=nml_on)  for img in image_val] 

        elif image_type == InputFeature.EMB3:
            image_train = [RFilter.emboss(img, FilterKernel.EMB33, nml_on=nml_on) for img in image_train]
            image_test  = [RFilter.emboss(img, FilterKernel.EMB33, nml_on=nml_on)  for img in image_test] 
            if self.image_val is not None:
                image_val  = [RFilter.emboss(img, FilterKernel.EMB33, nml_on=nml_on)  for img in image_val] 

        elif image_type == InputFeature.KNN1:
            image_train = [RFilter.knn_ave(img, 3, 3, nml_on=nml_on) for img in image_train]
            image_test  = [RFilter.knn_ave(img, 3, 3, nml_on=nml_on)  for img in image_test] 
            if self.image_val is not None:
                image_val  = [RFilter.knn_ave(img, 3, 3, nml_on=nml_on)  for img in image_val] 

        elif image_type == InputFeature.KNN2:
            image_train = [RFilter.knn_ave(img, 5, 5, nml_on=nml_on) for img in image_train]
            image_test  = [RFilter.knn_ave(img, 5, 5, nml_on=nml_on)  for img in image_test] 
            if self.image_val is not None:
                image_val  = [RFilter.knn_ave(img, 5, 5, nml_on=nml_on)  for img in image_val] 

        elif image_type == InputFeature.BLT1:
            image_train = [RFilter.bilateral(img, 3, 1.0, 2.0, nml_on=nml_on) for img in image_train]
            image_test  = [RFilter.bilateral(img, 3, 1.0, 2.0, nml_on=nml_on)  for img in image_test] 
            if self.image_val is not None:
                image_val  = [RFilter.bilateral(img, 3, 1.0, 2.0, nml_on=nml_on)  for img in image_val] 

        elif image_type == InputFeature.BLT2:
            image_train = [RFilter.bilateral(img, 5, 1.0, 2.0, nml_on=nml_on) for img in image_train]
            image_test  = [RFilter.bilateral(img, 5, 1.0, 2.0, nml_on=nml_on)  for img in image_test] 
            if self.image_val is not None:
                image_val  = [RFilter.bilateral(img, 5, 1.0, 2.0, nml_on=nml_on)  for img in image_val] 

        elif image_type == InputFeature.OOO_:
            image_train = [RFilter.black(img) for img in image_train]
            image_test  = [RFilter.black(img)  for img in image_test] 
            if self.image_val is not None:
                image_val  = [RFilter.black(img)  for img in image_val] 

        else:
            print("image_type is wrong!")
            return

        # １つ１つを画像に戻してresultフォルダーに保存する
        os.makedirs(filtering_train_file, exist_ok=True)
        WriteFile.save_images(filtering_train_file, self.train_casenames, image_train)
        os.makedirs(filtering_test_file, exist_ok=True)
        WriteFile.save_images(filtering_test_file, self.test_casenames, image_test)

        imagenum_train = len(image_train)
        imagenum_test = len(image_test)
        imagesize = len(image_train[0])
        
        # numpyの形式に
        image_train = np.asarray([image_train], dtype=np.float64)
        image_train = np.asarray(image_train, dtype=np.float64).reshape((imagenum_train, imagesize, imagesize, 1))
        image_test = np.asarray([image_test], dtype=np.float64)
        image_test = np.asarray(image_test, dtype=np.float64).reshape((imagenum_test, imagesize, imagesize, 1))

        self.image_train = image_train
        self.image_test = image_test

        if self.image_val is not None:
            os.makedirs(filtering_val_file, exist_ok=True)
            WriteFile.save_images(filtering_val_file, self.val_casenames, image_test)

            imagenum_val = len(image_val)

            image_val = np.asarray([image_val], dtype=np.float64)
            image_val = np.asarray(image_val, dtype=np.float64).reshape((imagenum_val, imagesize, imagesize, 1))
            self.image_val = image_val
            return image_train, image_test, image_val
        else:
            return image_train, image_test

