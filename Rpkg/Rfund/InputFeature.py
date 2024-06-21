from enum import Enum

import numpy as np

class InputFeature(Enum):
    """
    フィルタの種類．詳細は藤中さんのワードファイル.
    
    """
    GRY_ = 'GRY_'
    NML1 = 'NML1'
    NML2 = 'NML2'
    NML3 = 'NML3'
    TOP1 = 'TOP1'
    TOP2 = 'TOP2'
    TOP3 = 'TOP3'
    TOP4 = 'TOP4'
    SBLX = 'SBLX'
    SBLY = 'SBLY'
    SBLM = 'SBLM'
    SBLD = 'SBLD'
    SBL1 = 'SBL1'
    SBL2 = 'SBL2'
    SBL3 = 'SBL3'
    SBL4 = 'SBL4'
    LPL1 = 'LPL1'
    LPL2 = 'LPL2'
    MEA1 = 'MEA1'
    MEA2 = 'MEA2'
    GAU1 = 'GAU1'
    GAU2 = 'GAU2'
    MED1 = 'MED1'
    MED2 = 'MED2'
    LBP1 = 'LBP1'
    LBP2 = 'LBP2'
    LBP3 = 'LBP3'
    ETC1 = 'ETC1'
    ETC2 = 'ETC2'
    STC1 = 'STC1'
    STC2 = 'STC2'
    HGF_ = 'HGF_'
    NGP_ = 'NGP_'
    POS1 = 'POS1'
    POS2 = 'POS2'
    POS3 = 'POS3'
    SOL_ = 'SOL_'
    EMB1 = 'EMB1'
    EMB2 = 'EMB2'
    EMB3 = 'EMB3'
    KNN1 = 'KNN1'
    KNN2 = 'KNN2'
    BLT1 = 'BLT1'
    BLT2 = 'BLT2'
    OOO_ = 'OOO_'
    ## Custom added features from now on

    #CANNY FILTER
    CAN1 = 'CAN1'
    CAN2 = 'CAN2'
    CAN3 = 'CAN3'

    #PREWITT FILTER
    PRE1 = 'PRE1'
    PRE2 = 'PRE2'
    PRE3 = 'PRE3'
    PRE4 = 'PRE4'

    #UNSHARPENING FILTER
    UNS1 = 'UNS1'
    UNS2 = 'UNS2'
    UNS3 = 'UNS3'
    UNS4 = 'UNS4'
    UNS5 = 'UNS5'

    #FOURIER FILTER
    FOU1 = 'FOU1'
    FOU2 = 'FOU2'
    FOU3 = 'FOU3'
    FOU4 = 'FOU4'

    #EROSION FILTER
    ERO1 = 'ERO1'
    ERO2 = 'ERO2'
    ERO3 = 'ERO3'
    ERO4 = 'ERO4'
    ERO5 = 'ERO5'
    ERO6 = 'ERO6'

    #OPENING FILTER
    OPN1 = 'OPN1'
    OPN2 = 'OPN2'
    OPN3 = 'OPN3'
    OPN4 = 'OPN4'
    OPN5 = 'OPN5'

    #CLOSING FILTER
    CLO1 = 'CLO1'
    CLO2 = 'CLO2'
    CLO3 = 'CLO3'
    CLO4 = 'CLO4'
    CLO5 = 'CLO5'

    #BOX FILTER same as mean apparently so commented out
    # BOX1 = 'BOX1'
    # BOX2 = 'BOX2'
    # BOX3 = 'BOX3'
    # BOX4 = 'BOX4'

    #SCHARR GRADIENT FILTER
    SCH1 = 'SCH1'
    SCH2 = 'SCH2'
    SCH3 = 'SCH3'
    SCH4 = 'SCH4'
    
    #ROBERTS CROSS FILTER
    ROB1 = 'ROB1'
    ROB2 = 'ROB2'
    ROB3 = 'ROB3'
    ROB4 = 'ROB4'

    #MIN FILTER 
    MIN1 = 'MIN1'
    MIN2 = 'MIN2'
    MIN3 = 'MIN3'
    MIN4 = 'MIN4'
    
    #MAX FILTER
    MAX1 = 'MAX1'
    MAX2 = 'MAX2'
    MAX3 = 'MAX3'
    MAX4 = 'MAX4'

    #MORPH GRADIENT FILTER
    MRG1 = 'MRG1'
    MRG2 = 'MRG2'
    MRG3 = 'MRG3'
    MRG4 = 'MRG4'

    #MORPH LAPLACIAN FILTER
    MRL1 = 'MRL1'
    MRL2 = 'MRL2'
    MRL3 = 'MRL3'
    MRL4 = 'MRL4'

    # BOTTOM HAT TRANS
    BTM1 = 'BTM1'
    BTM2 = 'BTM2'
    BTM3 = 'BTM3'
    BTM4 = 'BTM4'

    #DISTANCE TRANSFORM
    DST_ = 'DST_'

    #HOMOMORPHIC FILTER
    HOM_ = 'HOM_'

    #STRUCTURE TENSOR FILTER
    # STR_ = 'STR_'

    #RICHARDSON LUCY FILTER
    RIC_ = 'RIC_'




    
    


    def __str__(self):
        return self.name

    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self.value >= other.value
        return NotImplemented

    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented

    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented
        
    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented

def image_processing(image, feature):
    # get image size
    height, width, deepth = image.shape
    # prepare result image
    f_img = np.zeros((height, width, deepth))
    # each image pre-processing
    if feature == InputFeature.GRY_:
        f_img = image
    
    return f_img