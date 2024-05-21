import os
import random
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.optimizers import Adam

from Rpkg.Rfund.InputFeature import InputFeature
from Rpkg.Rfund import ReadFile, WriteFile
from Rpkg.Rmodel import Unet, Mnet

from MCtool import RFilter, resultEval

"""
[注意] これはモジュールです．
"""

def set_seed(seed=7):
    """
    乱数シードseedで乱数固定
    """
    os.environ['PYTHONHASHSEED'] = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = ""

    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)


# 二値化処理(判別分析法)
def threshold_otsu(image, min_value=0, max_value=255):
    """
    濃淡画像を二値化する．
    :param image: 入力画像 
    :param min: 二値の小さい方(デフォルト: 0)
    :param max: 二値の大きい方(デフォルト: 255)

    :returns: 二値画像
    """
    if len(image.shape) == 3:
        height, width, _ = image.shape
        image = np.asarray(image, dtype=np.uint8).reshape((height, width))
    else:
        height, width = image.shape

    # ヒストグラムの算出
    hist = [np.sum(image == i) for i in range(256)]

    s_max = (0,-10)

    for th in range(256):
        
        # クラス1とクラス2の画素数を計算
        n1 = sum(hist[:th])
        n2 = sum(hist[th:])
        
        # クラス1とクラス2の画素値の平均を計算
        if n1 == 0 : mu1 = 0
        else : mu1 = sum([i * hist[i] for i in range(0,th)]) / n1   
        if n2 == 0 : mu2 = 0
        else : mu2 = sum([i * hist[i] for i in range(th, 256)]) / n2

        # クラス間分散の分子を計算
        s = n1 * n2 * (mu1 - mu2) ** 2

        # クラス間分散の分子が最大のとき、クラス間分散の分子と閾値を記録
        if s > s_max[1]:
            s_max = (th, s)
    
    # クラス間分散が最大のときの閾値を取得
    t = s_max[0]

    # 算出した閾値で二値化処理
    image[image < t] = min_value
    image[image >= t] = max_value

    return image


def save_eval_result(predict_image, gt_image, test_filenames, resDir, mask_image=None, overlay_on=False, ori_image=None):
    """
    予測画像に対する評価を行い，後処理(ここでは二値化のみ)を施した結果画像，
    TP，FP，FNを視覚化した結果画像，数値評価(Precision,Recall,F値)をしたCSVファイルを出力する．
    それぞれresDirの中に作成されます．

    :param predict_image: 予測画像(numpy形式のリスト) 
    :param gt_image: 正解画像(numpy形式のリスト)
    :param test_filnames: 画像のファイル名
    :param resDir: 結果フォルダのパス
    :param mask_image: マスク画像がある場合マスク画像があるパスを入力
    :param overlay_on: 重ね合わせ結果も出力するときTrue(デフォルト: False)
    :param ori_image: 原画像(numpy形式のリスト)
    
    """
    # 後処理後の画像フォルダ
    resBinDir = resDir + 'postProc-predict/'
    os.makedirs(resBinDir, exist_ok=True)

    # 評価画像のフォルダ
    evalDir = resDir + 'evaluation/'
    os.makedirs(evalDir, exist_ok=True)

    if overlay_on:
        # 評価画像のフォルダ
        overlayDir = resDir + 'overlay/'
        os.makedirs(overlayDir, exist_ok=True)

    # 評価結果のCSV
    evalTotalCsv = resDir + 'evaluation.csv'

    # もしSACsvはすでに作られているファイルならここは実行されない
    if not os.path.exists(evalTotalCsv):
        eva_total_df = pd.DataFrame(index=['case'], columns=['precision', 'recall', 'F-measure'])
        eva_total_df.to_csv(evalTotalCsv)

    eva_total_df = pd.read_csv(evalTotalCsv, index_col=0)
    eva_total_df.to_csv(evalTotalCsv)

    binImgs = [threshold_otsu(img) for img in predict_image]

    # precision, recall, F値の平均を求める
    meanPrecision = 0.0
    meanRecall = 0.0
    meanFmeasure = 0.0

    # 評価計算
    for i in range(0, len(binImgs)):
        # 予測された確率画像について後処理を施す
        # ここでは単に二値化する
        binImg = binImgs[i]
        cv2.imwrite(resBinDir + test_filenames[i], binImg)

        # 正解画像の読み込み
        # gt_img_path = gtDir + test_filenames[i]
        gtImg = gt_image[i]

        # マスク画像がある場合にはマスク外の結果を除外する
        if (mask_image is not None):
            ysize, xsize = binImg.shape[:2]
            for y in range(0,ysize):
                for x in range(0,xsize):
                    if mask_image[i][y][x] == 0:
                        binImg[y][x] = 0
        

        # TP/FP/FNの色分け
        precision, recall, Fmeasure, evalImg = resultEval.compare_res_lab(binImg, gtImg)
        cv2.imwrite(evalDir + test_filenames[i], evalImg)

        if overlay_on:
            oriImg = ori_image[i]
            overlayImg = resultEval.overlay_eval(binImg, oriImg, gtImg)
            cv2.imwrite(overlayDir + test_filenames[i], overlayImg)

        # CSVに結果の書き込み
        case_name = test_filenames[i].split('.')[0]
        eva_total_df.loc[case_name] = [precision, recall, Fmeasure]

        # 平均計算途中
        meanPrecision += precision
        meanRecall += recall
        meanFmeasure += Fmeasure

    # 平均計算
    #meanPrecision /= len(binImgs)
    #meanRecall /= len(binImgs)  
    #meanFmeasure /= len(binImgs)

    # CSVに平均の結果を書き込み
    eva_total_df.loc['Avearage'] = [meanPrecision, meanRecall, meanFmeasure]
    eva_total_df.to_csv(evalTotalCsv)

    return meanPrecision, meanRecall, meanFmeasure

class LearnAndTest():
    """
    学習とテストを行うクラス．現在U-NetとM-Net向けにしか書いていない．
    次のコードをこのクラスを使用する前にいれておくといいでしょう．
    
    ```
    # GPUメモリの指定
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        for device in physical_devices:
            tf.config.experimental.set_virtual_device_configuration(
            device,
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*8)])
            print('{} memory growth: {}'.format(device, tf.config.experimental.get_memory_growth(device)))
    else:
        print("Not enough GPU hardware devices available")

    # ログにエラーを出力
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    ```

    :examples:
    ```
    dnn = LearnAndTest()
    dnn.learning(model, image_train, label_train, image_val, label_val, EPOCHS, BATCH_SIZE)
    dnn.test(image_test, label_test, image_test_filenames, resDir)
    ```
    """
    # コンストラクタ
    def __init__(self) -> None:
        self.model = None
        pass

    def learning(self, model, image_train, label_train, image_val=None, label_val=None, EPOCHS=100, BATCH_SIZE=1):
        """
        学習を行う. 
        
        !!!注意!!!    画像は0-1正規化を行っていること．
        :param model: 構築したネットワークのモデル(例: model = Mnet.build_network(IMAGE_SIZE,input_channel=1)  )
        :param image_train, label_train: 学習データ．それぞれ原画像と対応するラベル画像
        :param image_val, label_val: 検証データ．それぞれ原画像と対応するラベル画像．デフォルトはNone．
        :param EPOCHS: 学習のエポック数．デフォルトは100．
        :param BATCH_SIZE: バッチサイズ．デフォルトは1

        :returns model 学習済みモデル     
        """
        if image_val is None:
            val_data = None
        else:
            val_data = (image_val, label_val)

        set_seed()# 乱数固定
        model.compile(
            loss='binary_crossentropy',
            optimizer=Adam(lr=0.001),
            metrics=['accuracy']
            )   # 損失関数と最適化関数の設定

        set_seed()# 乱数固定
        training = model.fit(
            image_train, label_train,
            epochs=EPOCHS, 
            batch_size=BATCH_SIZE,   
            validation_data=val_data,
            # use_multiprocessing=True,     
            shuffle=False,
            verbose=1
            )   # fit model 
        # plot_history(training)
    
        self.model = model
        return model

    def test(self, image_test, label_test, test_filenames, resDir, model=None, mask_image=None, overlay_on=False, ori_image=None):
        """
        テストを行う.  

        !!!注意!!!    画像は0-1正規化を行っていること．
        :param image_test, label_test: テストデータ．それぞれ原画像と対応するラベル画像
        :param test_filenames: テストデータのファイル名．
        :param resDir: 出力先のディレクトリ．ここには結果の確率画像が入った'predit'，
            それに対し二値化処理を施した画像が入った'postProc-predict'，
            TP/FP/FNの色分けされた画像が入った'evaluation'，
            precision, recall, F値の結果が出力された'evaluation.csv'が出力される．
        :param mask_image: マスク画像がある場合マスク画像があるパスを入力
        :param model: 学習済みモデル．デフォルトは直前に学習したモデル

        """
        # 学習から引き続きか学習済みモデルを使用するか
        if model is None:
            # results = self.model(image_test)
            results = self.model.predict(image_test, verbose=1)
        else:
            # results = model(image_test)
            results = model.predict(image_test, verbose=1)

        imagesize = image_test.shape[0]

        # 0-1の範囲の値が出力されるので見やすいように255倍する
        label_test255 = label_test * 255.0
        results255 = results * 255.0  

        predictDir = resDir + 'predict/'
        os.makedirs(predictDir, exist_ok=True)

        # １つ１つを画像に戻してresultフォルダーに保存するよ
        WriteFile.save_images(predictDir, test_filenames, results255)

        # 予測フォルダを二値画像として読み込み
        # binImgs, _ = ReadFile.directory_images(predictDir, imagesize, 'Binary_OTSU')

        # 評価を行いファイルに保存
        meanPrecision, meanRecall, meanFmeasure = save_eval_result(predict_image=results255, 
                                                                    gt_image=label_test255, 
                                                                    test_filenames=test_filenames, 
                                                                    resDir=resDir,
                                                                    mask_image=mask_image,
                                                                    overlay_on = overlay_on,
                                                                    ori_image = ori_image
                                                                    )

        return meanPrecision, meanRecall, meanFmeasure