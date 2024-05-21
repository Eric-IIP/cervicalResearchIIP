# %%
from MCtool.RFilter import gray
from genericpath import exists
from matplotlib import image
import math
import sys
import time

import cv2
from matplotlib import pyplot as plt
from tensorflow.python.keras.backend import dtype
from DeepLearning import LearnAndTest
from Rpkg.Rfund.InputFeature import InputFeature
import datetime
import os
import gc
import tensorflow as tf
import random
import numpy as np
import pandas as pd

from Rpkg.Rfund import ReadFile, WriteFile
from Rpkg.Rmodel import Unet, Mnet

import Filtering

# %%
# GPUメモリの指定
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_virtual_device_configuration(
        device,
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*8)])    # ★★★ 1024*8はGPUに合わせて値を変更
        print('{} memory growth: {}'.format(device, tf.config.experimental.get_memory_growth(device)))
else:
    print("Not enough GPU hardware devices available")

# GPUメモリの確保
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# ログにエラーを出力
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# 自動ガベージコレクションを有効にする
gc.enable()

# %%
# 乱数固定
def set_seed(seed=7):
    tf.random.set_seed(seed)

    # for numpy.random
    np.random.seed(seed)
    # for built-in random
    random.seed(seed)
    # for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 計算順序の固定
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = 'true'

    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

# %%
def read_csv_image(rootpath, csvpath):
    """
    csvファイル内に書かれた名前の画像を，
    rootpathのディレクトリから探し，画像のリストを返す．
    
    :param rootpath: 画像が入っているフォルダのパス
    :param csvpath: 画像のファイル名が記述されたcsvファイルのパス
    
    :returns: imglist: 画像のリスト
    :returns: fnlist: ファイル名のリスト
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

# %%
# ★★★パラメータ設定
modelName = 'M-Net'  # 深層学習に使うネットワーク(M-Net or U-Net)
IMAGE_SIZE = 256   # 画像のサイズ(8の倍数にしてください)
EPOCHS = 100    # エポック数
BATCH_SIZE = 1    # 深層学習のバッチサイズ

CHANNEL_NUM = 3    # マルチチャンネル画像のチャンネル数指定
M_LAYER_NUM = 2    # 多層の層の数の指定
MCType = 'MC{0}'.format(M_LAYER_NUM)    # マルチチャンネル化の種類(数字は何層か)

EVAL_TEST_NUM = 10  # 最後の評価テストの回数

# %%
# ★★★実験データのベースディレクトリ，
baseDir = '../'   # ここにはデータセットや実験結果フォルダがある構造にしている

# ★★★入力ファイル設定
input_dir = baseDir + 'dataset/'    # データセットが諸々入っているディレクトリ

image_root_dir = input_dir + 'cropped-original/'   
image_dir = image_root_dir + 'image/'  # 入力画像が入っているディレクトリパス
label_dir = image_root_dir + 'label/'  # 入力画像のラベルが入っているディレクトリパス

group1_filenames = input_dir + 'input_info/group1.csv'    # group1の一覧のcsvファイルのパス
group2_filenames = input_dir + 'input_info/group2.csv'      # group2の一覧のcsvファイルのパス
group3_filenames = input_dir + 'input_info/group3.csv'     # group3の一覧のcsvファイルのパス

# ★★★出力ファイル設定 
output_dir = baseDir + 'experiment/'   

# 現在の日付の取得
now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
date = now.strftime('%Y%m%d')     

# 結果フォルダの作成
expName = date + '_' + modelName + '_' + MCType + '_{0}ch'.format(CHANNEL_NUM) 
resDir = WriteFile.make_folder(output_dir, expName, cmd_mode=True) # コマンドライン上で対話ありに注意

#%%
# 画像データ->numpy形式に変換
image_g1, image_g1_filenames = read_csv_image(image_dir, group1_filenames)
label_g1, label_g1_filenames = read_csv_image(label_dir, group1_filenames)
image_g2, image_g2_filenames = read_csv_image(image_dir, group2_filenames)
label_g2, label_g2_filenames = read_csv_image(label_dir, group2_filenames)
image_g3, image_g3_filenames = read_csv_image(image_dir, group3_filenames)
label_g3, label_g3_filenames = read_csv_image(label_dir, group3_filenames)

# %%
def get_group_name(group_num):
    """
    グループの名前を取得
    
    :param group_num: グループの番号(ここでは1~3)
    
    :returns:  filenames: csvファイルのパス
    :returns: name: グループの名前
    """
    if group_num == 1:
        filenames = group1_filenames
    elif group_num == 2:
        filenames = group2_filenames
    elif group_num == 3:
        filenames = group3_filenames
    else: 
        print('<<エラー>> get_group_nameには1〜3を入力します．')
        return

    fn_a = filenames.split('/')
    fn = fn_a[len(fn_a)-1].split('.')
    name = fn[0] 
    return filenames, name

def get_image_data(group_num):
    """
    画像を取得
    
    :param group_num: グループの番号(ここでは1~3)
    
    :returns:  image: 画像numpyのlist
    :returns: image_filnames: ファイル名
    """
    if group_num == 1:
        image = image_g1
        image_filenames = image_g1_filenames
    elif group_num == 2:
        image = image_g2
        image_filenames = image_g2_filenames
    elif group_num == 3:
        image = image_g3
        image_filenames = image_g3_filenames
    else: 
        print('<<エラー>> get_image_dataには1〜3を入力します．')
        return
    
    return image, image_filenames

def get_label_data(group_num):
    """
    ラベルを取得
    
    :param group_num: グループの番号(ここでは1~3)
    
    :returns:  label: 画像numpyのlist
    :returns: label_filnames: ファイル名
    """
    if group_num == 1:
        label = label_g1
        label_filenames = label_g1_filenames
    elif group_num == 2:
        label = label_g2
        label_filenames = label_g2_filenames
    elif group_num == 3:
        label = label_g3
        label_filenames = label_g3_filenames
    else: 
        print('<<エラー>> get_labe_dataには1〜3を入力します．')
        return
    
    return label, label_filenames


def check_single_filter(learn_group, test_group, nml_on):
    """
    単一フィルタで学習したときの結果のチェック．
    テストでの精度(F値)が大きい順にソートしたファイルを得ることができる．
    
    :param learn_group: 学習のグループの番号(ここでは1~3)
    :param test_group: テストのグループの番号(ここでは1~3)
        -nml_on: 標準化を行うか
          
    :returns:  ini_filter_df: 単一フィルタの結果
    """
    train_filenames, train_name = get_group_name(learn_group)
    test_filenames, test_name = get_group_name(test_group)

    label_train, label_train_filenames = get_label_data(learn_group)
    label_test, label_test_filenames = get_label_data(test_group)

    # ラベル画像の0-1正規化
    label_train /= np.max(label_train)
    label_test /= np.max(label_test)

    # 単一フィルタで学習したときの結果(F値)ファイル
    ini_filter_csvpath =  resDir + 'single_filter_learn{0}_test{1}_result.csv'.format(train_name, test_name)
    if not os.path.exists(ini_filter_csvpath):
        ini_filter_df = pd.DataFrame(index=['filter_name'], columns=['Precision', 'Recall', 'F-measure'])
        ini_filter_df.to_csv(ini_filter_csvpath)
    ini_filter_df = pd.read_csv(ini_filter_csvpath, index_col=0)

    # 入力1chでの学習
    print("単一フィルタ画像で学習を始めます．")

    # テスト結果を入れるフォルダ
    test_resDir = resDir + 'learn{0}_test{1}/single_filter/'.format(train_name, test_name)
    os.makedirs(test_resDir, exist_ok=True)

    # 特徴画像を保存するフォルダ
    if nml_on:
        nml_on_str = 'nml_on/'
    else:
        nml_on_str = 'nml_off/'
    featureRootDir = input_dir + 'feature-dataset/ftimes_1/' + nml_on_str
    os.makedirs(featureRootDir, exist_ok=True)

    for inputfeature in InputFeature:
        # 特徴画像フォルダ名
        feature_name = inputfeature.value
        featureDir = featureRootDir + feature_name + '/'

        if feature_name in ini_filter_df.index.values:
            print('学習済みのため{0}の実行をスキップします．'.format(feature_name))
            continue    # 既に学習済みなら飛ばす

        featureDir_train = featureDir + train_name
        featureDir_test = featureDir + test_name
        if os.path.exists(featureDir_train) and os.path.exists(featureDir_test):
            image_train, _ = ReadFile.directory_images(featureDir + train_name, IMAGE_SIZE, 'Gray')
            image_test, _ = ReadFile.directory_images(featureDir + test_name, IMAGE_SIZE, 'Gray')

        else:
            os.makedirs(featureDir, exist_ok=True)
            image_train, image_train_filenames = read_csv_image(image_dir, train_filenames)
            image_test, image_test_filenames = read_csv_image(image_dir, test_filenames)
            image_train = Filtering.list_images(image_train, inputfeature, nml_on=True)
            image_test = Filtering.list_images(image_test, inputfeature, nml_on=True)
            # １つ１つを画像に戻してresultフォルダーに保存する
            os.makedirs(featureDir_train, exist_ok=True)
            WriteFile.save_images(featureDir_train, image_train_filenames, image_train)
            os.makedirs(featureDir_test, exist_ok=True)
            WriteFile.save_images(featureDir_test, image_test_filenames, image_test)

        # 画素値0-1正規化
        if inputfeature is not InputFeature.OOO_:
            image_train /= np.max(image_train)
            image_test /= np.max(image_test)
        else:
            image_train = np.asarray(image_train, dtype=np.uint8)
            image_test = np.asarray(image_test, dtype=np.uint8)

        # 乱数固定
        set_seed()   

        # ネットワーク構築  
        if modelName == 'M-Net':
            model = Mnet.build_network(IMAGE_SIZE,input_channel=1)  # M-Netネットワーク構築
        elif modelName == 'U-Net':
            model = Unet.build_network(IMAGE_SIZE,input_channel=1)  # U-Netネットワーク構築
        else:
            print('<<警告>> モデル名が間違っています．')
            print('学習とテストをスキップします．')
            break

        dnn = LearnAndTest()    # 学習とテストの準備

        # 学習を行う
        print("{0}について学習".format(inputfeature))
        dnn.learning(model, image_train, label_train, None, None, EPOCHS, BATCH_SIZE)

        #  テストを行う
        print("{0}についてテスト".format(inputfeature))
        eval_resDir = test_resDir + feature_name + '/'
        os.makedirs(eval_resDir, exist_ok=True)
        meanPrecision, meanRecall, meanFmeasure = dnn.test(image_test, label_test, label_test_filenames, eval_resDir)

        # 単一フィルタの結果への書き込み
        ini_filter_df.loc[feature_name] = [meanPrecision, meanRecall, meanFmeasure]
        ini_filter_df.to_csv(ini_filter_csvpath)
        del dnn

    print("<<成功>> 単一フィルタ画像でのすべての学習が終わりました．")

    # F値でソートする(降順)
    ini_filter_df = ini_filter_df.sort_values('F-measure', ascending=False, na_position='first')
    ini_filter_df.to_csv(ini_filter_csvpath)

    return ini_filter_df

# %%
# マルチチャンネル化
# マルチチャンネル画像の作成
# マルチチャンネル画像を保存するを付け加えてもいいが今はとりあえず放置
def create_multi_channel(feature_array, group_num):
    """
    マルチチャンネル画像を作成する．
    
    :param feature_array: 特徴画像のリスト
    :param group_num: グループの番号(ここでは1~3)
          
    :returns:  multi_channel_images: マルチチャンネルの画像(numpy)のリスト
    """

    featureRootDir = input_dir + 'feature-dataset/'

    _, group_name = get_group_name(group_num)
    image_list, file_name_list = get_image_data(group_num)

    channel_num = len(feature_array)
    ftimes = len(feature_array[0])
    
    def do_filtering(image, filename, image_type_list, times_count=0):
        """
        フィルタリング処理を施す．
        
        :param iamge: 画像(numpy形式，サイズはn×mの形にすること)
        :param filename: 画像のファイル名(◯◯.pngなどの拡張子がついている形にすること)
        :param image_type_list: フィルタリングの種類リスト
        :param times_count: フィルタリングした回数(デフォルト:0)
        
        :returns: image: image_type_listの中身でフィルタリングした画像
        """
        nml_str = 'nml_off'
        if times_count >= ftimes:
            # フィルタリング終了
            return image
        elif times_count == ftimes - 1:
            nml_on = True
            nml_str = 'nml_on'
        else:
            nml_on = False
        image_type = image_type_list[times_count]
        # print(image_type)

        feature_name = ''
        for i in range(0, times_count+1):
            feature_name = feature_name + image_type_list[i].value

        featureDir = featureRootDir + 'ftimes_{0}/{1}/'.format(times_count+1, nml_str)
        featureDir = featureDir + feature_name + '/' + group_name + '/'
        os.makedirs(featureDir, exist_ok=True)
        featureImgPath = featureDir + filename

        # 特徴画像が作成済みならそこからとる
        if os.path.exists(featureImgPath):
            f_img = cv2.imread(featureImgPath, cv2.IMREAD_GRAYSCALE)
        # 作成済みでないなら生成してファイルに保存する
        else: 
            f_img = Filtering.single_image(image, image_type, nml_on)
            cv2.imwrite(featureImgPath, f_img)

        return do_filtering(f_img, filename, image_type_list, times_count+1)

    multi_channel_images = []
    for i, img in enumerate(image_list):
        height, width, _ = img.shape
        img_dim0 = np.asarray(img, dtype=np.uint8).reshape((height, width))
        image_type_list = feature_array[0]
        f_img = do_filtering(img_dim0, file_name_list[i], image_type_list, 0)
        # マルチチャンネルにする
        multi_channel_image = f_img
        for j in range(1, CHANNEL_NUM):
            image_type_list = feature_array[j]
            f_img = do_filtering(img_dim0, file_name_list[i], image_type_list)
            # マルチチャンネルにする
            multi_channel_image = np.dstack([multi_channel_image, f_img])
            
        multi_channel_images.append(multi_channel_image)
                
    return multi_channel_images

# ----- ここから焼きなまし法 ----- #
# %%
def simulated_annealing(learn_group, test_group):
    """
    焼きなまし法の実行
    
    :param learn_group: 学習のグループの番号(ここでは1~3)
    :param test_group: テストのグループの番号(ここでは1~3)
          
    :returns:  ini_filter_df: 単一フィルタの結果
    :param multi_filter_df: マルチチャンネル化の結果
    """
    # 単一フィルタでの学習
    # if M_LAYER_NUM == 1:
    #     ini_nml_on = True
    # else:
    #     ini_nml_on = False
    # ini_filter_df = check_single_filter(learn_group, test_group, ini_nml_on)
    ini_filter_df = check_single_filter(learn_group, test_group, True)


    # 焼きなまし法のパラメータの初期化(初期状態の設定)
    feature_array = []    # 単一フィルタで精度が最も大きいものを同じチャンネルに入れる．
    feature_num_array = []    # 単一フィルタの結果の順番の番号，最初はすべて1位からスタート
    for i in range(0, CHANNEL_NUM):
        feature_array_c = []
        feature_num_array_c = []
        # 原画像を選ぶ
        for j in range(0, M_LAYER_NUM-1):
            feature_array_c.append(InputFeature.GRY_)
            feature_num_array_c.append(ini_filter_df.index.get_loc('GRY_'))    # GRY_の位置取得
        # 最後に1位のフィルタを選択
        image_type = InputFeature(ini_filter_df.index.values[1])
        feature_array_c.append(image_type)
        feature_num_array_c.append(1)    # 1の意味: 1位
        
        feature_array.append(feature_array_c)
        feature_num_array.append(feature_num_array_c)

    # ラベル画像の取得
    label_train, label_train_filenames = get_label_data(learn_group)
    label_test, label_test_filenames = get_label_data(test_group)

    # ラベル画像の0-1正規化
    label_train /= np.max(label_train)
    label_test /= np.max(label_test)

    # ★★★焼きなましのパラメータ設定
    maxT = 500    # 最大温度
    cool = 0.9  # 冷却係数
    stepMax = int(len(InputFeature) / 2) + 1   # ジャンプ範囲(最大)
    acceptCountMax = 100    # 最大アクセプト回数

    # 結果の初期化
    index = 1   # 現在のインデックス
    t = maxT    # 現在の温度
    step = stepMax    # 現在のジャンプ範囲
    cost = sys.float_info.max   # コスト
    acceptCount = int(0)    # アクセプト回数
    rejectCount = int(0)    # リジェクト回数

    # 結果の出力先
    _, train_name = get_group_name(learn_group)
    _, test_name = get_group_name(test_group)
    multi_resDir = resDir + 'learn{0}_test{1}/multi_filter/'.format(train_name, test_name)
    os.makedirs(multi_resDir, exist_ok=True)
    multi_filter_csvpath =  resDir + '{0}_SA_T{0}_cool{1}_stepMax{2}_learn{3}_test{4}_result.csv'.format(maxT, cool, stepMax, train_name, test_name)

    if not os.path.exists(multi_filter_csvpath):
        multi_filter_df = pd.DataFrame(index=['index'], columns=['acceptCount', 'rejectCount', 'T', 'step'])
        # 各チャンネルの列を作成
        for i in range(0, CHANNEL_NUM):
            for j in range(0, M_LAYER_NUM):
                column_name = 'CH{0}_F{1}_filter'.format(i+1, j+1)
                multi_filter_df[column_name] = np.NaN
            for j in range(0, M_LAYER_NUM):
                column_name = 'CH{0}_F{1}_rank'.format(i+1, j+1)
                multi_filter_df[column_name] = np.NaN
        
        # その他の結果列追加
        multi_filter_df['F-measure'] = np.NaN
        multi_filter_df['cost'] = np.NaN
        multi_filter_df['explanation'] = np.NaN
        multi_filter_df['result_file'] = np.NaN
        multi_filter_df['time'] = np.NaN

    else:
        multi_filter_df = pd.read_csv(multi_filter_csvpath, index_col=0) 
        # 既に書き込み済みデータがあればパラメータを変更
        index_num = len(multi_filter_df.index)
        if  index_num > 1:
            index = multi_filter_df.index[index_num-1]
            acceptCount = int(multi_filter_df.loc[index]['acceptCount'])    # アクセプト回数
            rejectCount = int(multi_filter_df.loc[index]['rejectCount'])    # リジェクト回数
            t = float(multi_filter_df.loc[index]['T']) * cool   # 記録された温度から下げる
            step = int(multi_filter_df.loc[index]['step'])    # 記録されたジャンプ範囲
            cost =  float(multi_filter_df.loc[index]['cost'])  # コスト

            for i in range(0, CHANNEL_NUM):
                for j in range(0, M_LAYER_NUM):
                    column_name = 'CH{0}_F{1}_filter'.format(i+1, j+1)
                    feature_array[i][j] = InputFeature(multi_filter_df.loc[index][column_name])
                for j in range(0, M_LAYER_NUM):
                    column_name = 'CH{0}_F{1}_rank'.format(i+1, j+1)
                    feature_num_array[i][j] = int(multi_filter_df.loc[index][column_name])

            index = int(index) + 1 # 数値に変える

            print('{0}回目，温度{1}から焼きなましを再開します．'.format(index, t))

    # CSVを出力
    multi_filter_df.to_csv(multi_filter_csvpath)

    # 作成されたCSVファイルの読み込み
    multi_filter_df = pd.read_csv(multi_filter_csvpath, index_col=0)  

    # 焼きなまし法でフィルタの組み合わせを探す
    while acceptCount < acceptCountMax:
        t_multi_start = time.time()
        # 実行箇所の表示
        progress_percent = '{:.3f}'.format(acceptCount*100/acceptCountMax)
        print('{0}回目，温度{1}の焼きなまし開始．(進行 {2}%)'.format(index, t, acceptCount, progress_percent))

        # stepの値を徐々に小さくする
        stepKeep = stepMax * t / maxT
        if stepKeep > 1:
            step = round(stepKeep)
        else:
            step = 1
        print('-- [1/4] stepが{0}で計算を始める．'.format(step))

        # 変数の値を変更する
        def change_channel():
            """
            変数の値を変更する．
            
            :param なし
            
            :returns: new_feature_array: 変更した特徴リスト 
            :returns: new_feature_num_array: 変更した特徴リスト(番号)
            """
            new_feature_array = [[0] * M_LAYER_NUM for i in range(CHANNEL_NUM)]
            new_feature_num_array = [[0] * M_LAYER_NUM for i in range(CHANNEL_NUM)]

            # 変えるフィルタの種類をランダムに変更する
            for i in range(0, CHANNEL_NUM):
                for j in range(0, M_LAYER_NUM):
                    new_ch_num = -1
                    while (new_ch_num <= 0) or (new_ch_num > len(InputFeature)):
                        new_ch_num = feature_num_array[i][j] + random.randint(-step, step)
                    new_feature_num_array[i][j] = int(new_ch_num)
                    image_type = InputFeature(ini_filter_df.index.values[new_ch_num])
                    new_feature_array[i][j] = image_type

            # Channelの順番が変わっても対応させるため，ソートする関数をつくる
            new_feature_num_array = sorted(new_feature_num_array)

            # ここで注意することは単一順位と対応させてnew_feature_arrayもソート
            for i in range(0, CHANNEL_NUM):
                for j in range(0, M_LAYER_NUM):
                    rank = int(new_feature_num_array[i][j])
                    new_feature_array[i][j] = InputFeature(ini_filter_df.index.values[rank])

            return new_feature_array, new_feature_num_array

        # 既にファイルが存在しているかチェックする(焼きなまし結果のCSVファイルから)
        def check_already_exists(folder_name):
            already_exists = False  
            index_num = len(multi_filter_df.index)
            for i in range(1, index_num):
                if multi_filter_df.loc[multi_filter_df.index[i]]['result_file'] == folder_name:
                    already_exists = True
                    break
            return already_exists

        print('-- [2/4] 仮に特徴画像の組み合わせを変更する．'.format(step))
        
        multi_folder_path = ''    
        multi_folder_name = ''
        
        for i in range(0, CHANNEL_NUM-1):
            for j in range(0, M_LAYER_NUM):
                multi_folder_name = multi_folder_name + feature_array[i][j].value 
            multi_folder_name = multi_folder_name + '_'

        for j in range(0, M_LAYER_NUM):
            multi_folder_name = multi_folder_name + feature_array[CHANNEL_NUM-1][j].value 
        multi_folder_path = multi_resDir + multi_folder_name + '/'

        # 最初は単一フィルタ最上位のものからスタート
        if index == 1:
            os.makedirs(multi_folder_path, exist_ok=True)
            new_feature_array = [[0] * M_LAYER_NUM for i in range(CHANNEL_NUM)]
            new_feature_num_array = [[0] * M_LAYER_NUM for i in range(CHANNEL_NUM)]
            for i in range(0, CHANNEL_NUM):
                for j in range(0, M_LAYER_NUM):
                    new_feature_array[i][j] = feature_array[i][j]
                    new_feature_num_array[i][j] = int(feature_num_array[i][j])

        # 順序関係なくすでに存在しているものなら学習を飛ばす
        # 今回の焼きなましで既に存在しているかをチェック
        # ダブリチェックなしを書き換える ★★★★
        while check_already_exists(multi_folder_name):
            multi_folder_name = ''
            new_feature_array, new_feature_num_array = change_channel()
            for i in range(0, CHANNEL_NUM-1):
                for j in range(0, M_LAYER_NUM):
                    multi_folder_name = multi_folder_name + new_feature_array[i][j].value 
                multi_folder_name = multi_folder_name + '_'
            for j in range(0, M_LAYER_NUM):
                multi_folder_name = multi_folder_name + new_feature_array[CHANNEL_NUM-1][j].value 
        
        multi_folder_path = multi_resDir + multi_folder_name + '/'
        os.makedirs(multi_folder_path, exist_ok=True)
    
        print('-- [3/4] 仮の組み合わせについて学習とテストする．')
        print('-------- マルチチャンネル画像作成中.....．')
        # マルチチャンネル画像の作成
        multi_channel_image_train = create_multi_channel(new_feature_array, learn_group)
        multi_channel_image_test = create_multi_channel(new_feature_array, test_group)

        # 画素値0-1正規化
        multi_channel_image_train /= np.max(multi_channel_image_train)
        multi_channel_image_test /= np.max(multi_channel_image_test)

        # 乱数固定
        set_seed()   

        # ネットワーク構築  
        if modelName == 'M-Net':
            model = Mnet.build_network(IMAGE_SIZE,input_channel=CHANNEL_NUM)  # M-Netネットワーク構築
        elif modelName == 'U-Net':
            model = Unet.build_network(IMAGE_SIZE,input_channel=CHANNEL_NUM)  # U-Netネットワーク構築
        else:
            print('-------- <<エラー>> モデル名が間違っています．')
            print('-------- 焼きなましを終了します．')
            break

        dnn = LearnAndTest()    # 学習とテストの準備

        # 学習を行う
        print("{0}について学習".format(multi_folder_name))
        dnn.learning(model, multi_channel_image_train, label_train, None, None, EPOCHS, BATCH_SIZE)

        #  テストを行う
        print("{0}についてテスト".format(multi_folder_name))
        meanPrecision, meanRecall, meanFmeasure = dnn.test(multi_channel_image_test, label_test, label_test_filenames, multi_folder_path)     

        print('-- [4/4] 仮の組み合わせの採用するか確認する．')
        new_cost = 1 - meanFmeasure
        # warningがでるから書き換える
        if str(index) in multi_filter_df.values:
           pass   # 何もしない
        else:
            multi_filter_df.loc[str(index)] = '-' 
        # 確率の計算
        if acceptCount > 0:
            try:
                p = math.pow(math.e, (cost-new_cost)/t)
            except OverflowError:
                multi_filter_df.loc[str(index)]['explanation'] = 'p_is_overflow'
                p = 1
        else:
            p = sys.float_info.max

        # 変更後のコストが小さければ採用する
        # コストが大きい場合は確率的に採用する
        if (new_cost < cost) or random.random() < p: 
            print('-------- 採用．')

            acceptCount += 1
            # CSVファイルへの書き出し
            multi_filter_df.loc[str(index)]['acceptCount'] = int(acceptCount)
            multi_filter_df.loc[str(index)]['rejectCount'] = int(rejectCount)
            multi_filter_df.loc[str(index)]['T'] = t
            multi_filter_df.loc[str(index)]['step'] = int(step)

            for i in range(0, CHANNEL_NUM):
                for j in range(0, M_LAYER_NUM):
                    column_name = 'CH{0}_F{1}_filter'.format(i+1, j+1)
                    multi_filter_df.loc[str(index)][column_name] = new_feature_array[i][j].value
                for j in range(0, M_LAYER_NUM):
                    column_name = 'CH{0}_F{1}_rank'.format(i+1, j+1)
                    multi_filter_df.loc[str(index)][column_name] = int(new_feature_num_array[i][j])
            
            multi_filter_df.loc[str(index)]['F-measure'] = meanFmeasure
            multi_filter_df.loc[str(index)]['cost'] = new_cost

            multi_filter_df.loc[str(index)]['result_file'] = multi_folder_name

            t_multi_end = time.time()  
            multi_filter_df.loc[str(index)]['time'] = t_multi_end - t_multi_start

            multi_filter_df.to_csv(multi_filter_csvpath)

            # 更新
            index += 1
            t = t * cool
            for i in range(0, CHANNEL_NUM):
                for j in range(0, M_LAYER_NUM):
                    feature_array[i][j] = new_feature_array[i][j]
                    feature_num_array[i][j] = new_feature_num_array[i][j]
            
            cost = new_cost

        else:
            print('-------- 棄却．')

            rejectCount += 1
            # CSVファイルへの書き出し
            multi_filter_df.loc[str(index)]['acceptCount'] = int(acceptCount)
            multi_filter_df.loc[str(index)]['rejectCount'] = int(rejectCount)
            multi_filter_df.loc[str(index)]['T'] = t
            multi_filter_df.loc[str(index)]['step'] = int(step)

            for i in range(0, CHANNEL_NUM):
                for j in range(0, M_LAYER_NUM):
                    column_name = 'CH{0}_F{1}_filter'.format(i+1, j+1)
                    multi_filter_df.loc[str(index)][column_name] = new_feature_array[i][j].value
                for j in range(0, M_LAYER_NUM):
                    column_name = 'CH{0}_F{1}_rank'.format(i+1, j+1)
                    multi_filter_df.loc[str(index)][column_name] = int(new_feature_num_array[i][j])

            multi_filter_df.loc[str(index)]['F-measure'] = meanFmeasure
            multi_filter_df.loc[str(index)]['cost'] = new_cost

            multi_filter_df.loc[str(index)]['explanation'] = 'not_adopted:do_not_use'
            multi_filter_df.loc[str(index)]['result_file'] = multi_folder_name

            t_multi_end = time.time()  
            multi_filter_df.loc[str(index)]['time'] = t_multi_end - t_multi_start

            multi_filter_df.to_csv(multi_filter_csvpath)

            # 更新
            index += 1

    print('<<成功>> 焼きなましを終了します．')
    return ini_filter_df, multi_filter_df

# %%
def eval_sa(ini_filter_df, multi_filter_df, test_sa_group, learn_group, test_group):
    """
    評価テストの実行
    
    :param ini_filter_df: 単一フィルタの結果
    :param multi_filter_df: マルチチャンネル化の結果
    :param test_sa_group: 焼きなましでのテストのグループの番号(ここでは1~3)
    :param learn_group: 学習のグループの番号(ここでは1~3)
    :param test_group: テストのグループの番号(ここでは1~3)
          
    :returns:  total_res_df: 原画像, 単一フィルタ，MC画像をまとめた結果
    """
    # グループ名の取得
    _, test_sa_name = get_group_name(test_sa_group)
    _, train_name = get_group_name(learn_group)
    _, test_name = get_group_name(test_group)

    # 原画像情報の取得
    image_train, _ = get_image_data(learn_group)
    image_test, _ = get_image_data(test_group)
    image_test_list, _ = get_image_data(test_group)

    # 原画像の0-1正規化
    image_train /= np.max(image_train)
    image_test /= np.max(image_test)

    # ラベル情報の取得
    label_train, _ = get_label_data(learn_group)
    label_test, label_test_filenames = get_label_data(test_group)

    # ラベル画像の0-1正規化
    label_train /= np.max(label_train)
    label_test /= np.max(label_test)

    # テスト結果を入れるフォルダ
    gray_resDir = resDir + 'learn{0}_test{1}/gray_test{2}/'.format(train_name, test_sa_name, test_name)

    # テスト結果を入れるフォルダ
    bset_single_f_type = InputFeature(ini_filter_df.index.values[1])
    single_resDir = resDir + 'learn{0}_test{1}/single_{2}_test{3}/'.format(train_name, test_sa_name, bset_single_f_type, test_name)

    # マルチチャンネル化で最もよかった組み合わせを読み込む
    sorted_multi_filter_df = multi_filter_df.sort_values('F-measure', ascending=False, na_position='first')
    index = sorted_multi_filter_df.iloc[1, 0]
    index = str(int(index))
    # 特徴リストの初期化
    feature_array = [[0] * M_LAYER_NUM for i in range(CHANNEL_NUM)]
    # 特徴リストをcsvファイルをとってくる
    for i in range(0, CHANNEL_NUM):
        for j in range(0, M_LAYER_NUM):
            column_name = 'CH{0}_F{1}_filter'.format(i+1, j+1)
            feature_array[i][j] = InputFeature(multi_filter_df.loc[index][column_name])
    combi_name = sorted_multi_filter_df.loc[index]['result_file']
    # テスト結果を入れるフォルダ
    test_resDir = resDir + 'learn{0}_test{1}/SA_{2}_test{3}/'.format(train_name, test_sa_name, combi_name, test_name)

    # グレースケールでテスト
    os.makedirs(gray_resDir, exist_ok=True)
    # 乱数固定
    set_seed()   
    # ネットワーク構築  
    if modelName == 'M-Net':
        model = Mnet.build_network(IMAGE_SIZE,input_channel=1)  # M-Netネットワーク構築
    elif modelName == 'U-Net':
        model = Unet.build_network(IMAGE_SIZE,input_channel=1)  # U-Netネットワーク構築
    else:
        print('-------- <<エラー>> モデル名が間違っています．')
        print('-------- 予測評価を終了します．')
    dnn = LearnAndTest()    # 学習とテストの準備
    # 学習を行う
    print("原画像について学習")
    dnn.learning(model, image_train, label_train, None, None, EPOCHS, BATCH_SIZE)
    #  テストを行う
    print("原画像についてテスト")
    dnn.test(image_test, label_test, label_test_filenames, gray_resDir, overlay_on=True, ori_image=image_test_list)

    del dnn

    # 単一で最高性能なフィルタでテスト
    os.makedirs(single_resDir, exist_ok=True)
    # 乱数固定
    set_seed()   
    # ネットワーク構築  
    if modelName == 'M-Net':
        model = Mnet.build_network(IMAGE_SIZE,input_channel=1)  # M-Netネットワーク構築
    elif modelName == 'U-Net':
        model = Unet.build_network(IMAGE_SIZE,input_channel=1)  # U-Netネットワーク構築
    else:
        print('-------- <<エラー>> モデル名が間違っています．')
        print('-------- 予測評価を終了します．')
    dnn = LearnAndTest()    # 学習とテストの準備
    # 学習を行う
    print("{0}について学習".format(bset_single_f_type))
    dnn.learning(model, image_train, label_train, None, None, EPOCHS, BATCH_SIZE)
    #  テストを行う
    print("{0}について学習".format(bset_single_f_type))
    dnn.test(image_test, label_test, label_test_filenames, single_resDir, overlay_on=True, ori_image=image_test_list)
    del dnn

    # 精度が最もよかった組み合わせについてテストを行う
    os.makedirs(test_resDir, exist_ok=True)
    # マルチチャンネル画像の作成
    multi_channel_image_train = create_multi_channel(feature_array, learn_group)
    multi_channel_image_test = create_multi_channel(feature_array, test_group) 
    # 画素値0-1正規化
    multi_channel_image_train /= np.max(multi_channel_image_train)
    multi_channel_image_test /= np.max(multi_channel_image_test)
    # 乱数固定
    set_seed()   
    # ネットワーク構築  
    if modelName == 'M-Net':
        model = Mnet.build_network(IMAGE_SIZE,input_channel=CHANNEL_NUM)  # M-Netネットワーク構築
    elif modelName == 'U-Net':
        model = Unet.build_network(IMAGE_SIZE,input_channel=CHANNEL_NUM)  # U-Netネットワーク構築
    else:
        print('-------- <<エラー>> モデル名が間違っています．')
        print('-------- 予測評価を終了します．')
    dnn = LearnAndTest()    # 学習とテストの準備
    # 学習を行う
    print("{0}について学習".format(combi_name))
    dnn.learning(model, multi_channel_image_train, label_train, None, None, EPOCHS, BATCH_SIZE)
    #  テストを行う
    print("{0}についてテスト".format(combi_name))
    dnn.test(multi_channel_image_test, label_test, label_test_filenames, test_resDir, overlay_on=True, ori_image=image_test_list) 
    del dnn

    # 結果をまとめる
    total_res_df = pd.DataFrame(index=['case'], columns=['','precision(G)', 'recall(G)', 'F-measure(G)', '', 'precision(S)', 'recall(S)', 'F-measure(S)', '', 'precision(M)', 'recall(M)', 'F-measure(M)'])
    total_res_df.iat[0, 0] = 'gray'
    total_res_df.iat[0, 4] = 'single'
    total_res_df.iat[0, 8] = 'multi'

    # 各結果を読み込む
    gray_res_df = pd.read_csv(gray_resDir + 'evaluation.csv')
    single_res_df = pd.read_csv(single_resDir + 'evaluation.csv')
    multi_res_df = pd.read_csv(test_resDir + 'evaluation.csv')

    for i in range(1, len(gray_res_df)-1):
        case_name = gray_res_df.iat[i, 0]
        
        total_res_df.at[case_name, 'precision(G)'] = gray_res_df.iat[i, 1]
        total_res_df.at[case_name, 'recall(G)']    = gray_res_df.iat[i, 2]
        total_res_df.at[case_name, 'F-measure(G)'] = gray_res_df.iat[i, 3]

        total_res_df.at[case_name, 'precision(S)'] = single_res_df.iat[i, 1]
        total_res_df.at[case_name, 'recall(S)']    = single_res_df.iat[i, 2]
        total_res_df.at[case_name, 'F-measure(S)'] = single_res_df.iat[i, 3]

        total_res_df.at[case_name, 'precision(M)'] = multi_res_df.iat[i,1]
        total_res_df.at[case_name, 'recall(M)']    = multi_res_df.iat[i,2]
        total_res_df.at[case_name, 'F-measure(M)'] = multi_res_df.iat[i,3]

    # total_res_df = pd.merge(gray_res_df, single_res_df, how='left')
    # total_res_df = pd.merge(total_res_df, multi_res_df, how='left')
    total_res_df.to_csv(resDir +'learn{0}_test{1}/total_evaluation.csv'.format(train_name, test_sa_name))

    return total_res_df

# %%
# ここからmain
# ★★★3-fold 3phase validationで実験を行う
LEARN_GROUP = [1, 2, 3]    # 学習グループは1,2,3の順
TEST_SA_GROUP = [2, 3, 1]    # 焼きなましのテストグループは2,3,1の順
TEST_GROUP = [3, 1, 2]    # 最終テストのテストグループは3,1,2の順

test_logs_Dir = resDir + 'test_logs/' 
os.makedirs(test_logs_Dir, exist_ok=True)

# テストをEVAL_TEST_NUM回だけ行う
for j in range(0, EVAL_TEST_NUM):
    total_res_df = pd.DataFrame(index=['case'], columns=['','precision(G)', 'recall(G)', 'F-measure(G)', '', 'precision(S)', 'recall(S)', 'F-measure(S)', '', 'precision(M)', 'recall(M)', 'F-measure(M)'])
    total_res_df.iat[0, 0] = 'gray'
    total_res_df.iat[0, 4] = 'single'
    total_res_df.iat[0, 8] = 'multi'
    for i in range(0,3):
        ini_filter_df, multi_filter_df = simulated_annealing(LEARN_GROUP[i], TEST_SA_GROUP[i])   # 焼きなましの実行(実行済みならスキップされる)
        res_df = eval_sa(ini_filter_df, multi_filter_df, TEST_SA_GROUP[i], LEARN_GROUP[i], TEST_GROUP[i])   # 結果のテスト
        
        # テスト結果をまとめる
        res_df = res_df.drop(res_df.index[0])
        total_res_df = pd.concat([total_res_df, res_df])
        total_res_df = total_res_df.sort_index()

    total_res_df.to_csv(test_logs_Dir +'total_evaluation{0}.csv'.format(j+1))

    del total_res_df

# EVAL_TEST_NUM回の平均算出する
# EVAL_TEST_NUM回の平均算出するファイル
total_res_df_n = pd.DataFrame(index=['case'], columns=['','precision(G)', 'recall(G)', 'F-measure(G)', '', 'precision(S)', 'recall(S)', 'F-measure(S)', '', 'precision(M)', 'recall(M)', 'F-measure(M)'])
total_res_df_n = pd.read_csv(test_logs_Dir +'total_evaluation{0}.csv'.format(1), index_col=0)
total_res_df_n.to_csv(resDir +'total_average_evaluation.csv')

for j in range(1, EVAL_TEST_NUM):
    total_res_df = pd.DataFrame(index=['case'], columns=['','precision(G)', 'recall(G)', 'F-measure(G)', '', 'precision(S)', 'recall(S)', 'F-measure(S)', '', 'precision(M)', 'recall(M)', 'F-measure(M)'])
    total_res_df = pd.read_csv(test_logs_Dir +'total_evaluation{0}.csv'.format(j+1), index_col=0)

    total_res_df_n['precision(G)'] = total_res_df_n['precision(G)'] + total_res_df['precision(G)']
    total_res_df_n['recall(G)']    = total_res_df_n['recall(G)'] + total_res_df['recall(G)']
    total_res_df_n['F-measure(G)'] = total_res_df_n['F-measure(G)'] + total_res_df['F-measure(G)']

    total_res_df_n['precision(S)'] = total_res_df_n['precision(S)'] + total_res_df['precision(S)']
    total_res_df_n['recall(S)']    = total_res_df_n['recall(S)'] + total_res_df['recall(S)']
    total_res_df_n['F-measure(S)'] = total_res_df_n['F-measure(S)'] + total_res_df['F-measure(S)']
    
    total_res_df_n['precision(M)'] = total_res_df_n['precision(M)'] + total_res_df['precision(M)']
    total_res_df_n['recall(M)']    = total_res_df_n['recall(M)'] + total_res_df['recall(M)']
    total_res_df_n['F-measure(M)'] = total_res_df_n['F-measure(M)'] + total_res_df['F-measure(M)']

total_res_df_n['precision(G)'] /= float(EVAL_TEST_NUM)
total_res_df_n['recall(G)']    /= float(EVAL_TEST_NUM)
total_res_df_n['F-measure(G)'] /= float(EVAL_TEST_NUM)

total_res_df_n['precision(S)'] /= float(EVAL_TEST_NUM) 
total_res_df_n['recall(S)']    /= float(EVAL_TEST_NUM)
total_res_df_n['F-measure(S)'] /= float(EVAL_TEST_NUM) 

total_res_df_n['precision(M)'] /= float(EVAL_TEST_NUM)
total_res_df_n['recall(M)']    /= float(EVAL_TEST_NUM)
total_res_df_n['F-measure(M)'] /= float(EVAL_TEST_NUM)

total_res_df_n.to_csv(resDir +'total_average_evaluation.csv')
# %%
