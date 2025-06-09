
import papermill as pm
import tempfile
import random
# Define your parameters

parameter_sets = [
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-GRY_'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-NML1'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-NML2'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-NML3'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-TOP1'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-TOP2'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-TOP3'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-TOP4'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-SBLX'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-SBLY'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-SBLM'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-SBLD'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-SBL1'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-SBL2'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-SBL3'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-SBL4'},
    
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-LPL1'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-LPL2'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-MEA1'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-MEA2'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-GAU1'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-GAU2'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-MED1'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-MED2'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-LBP1'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-LBP2'},
    
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-LBP3'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-ETC1'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-ETC2'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-STC1'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-STC2'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-HGF_'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-NGP_'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-POS1'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-POS2'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-POS3'},
    
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-SOL_'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-EMB1'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-EMB2'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-EMB3'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-KNN1'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-KNN2'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-BLT1'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-BLT2'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-OOO_'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-CAN1'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-CAN2'},
    
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-CAN3'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-PRE1'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-PRE2'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-PRE3'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-PRE4'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-UNS1'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-UNS2'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-UNS3'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-UNS4'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-UNS5'},
    
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-FOU1'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-FOU2'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-FOU3'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-FOU4'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-ERO1'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-ERO2'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-ERO3'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-ERO4'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-ERO5'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-ERO6'},
    
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-OPN1'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-OPN2'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-OPN3'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-OPN4'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-OPN5'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-CLO1'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-CLO2'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-CLO3'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-CLO4'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-CLO5'},
    
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-SCH1'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-SCH2'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-SCH3'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-SCH4'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-ROB1'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-ROB2'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-ROB3'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-ROB4'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-MIN1'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-MIN2'},
    
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-MIN3'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-MIN4'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-MAX1'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-MAX2'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-MAX3'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-MAX4'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-MRG1'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-MRG2'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-MRG3'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-MRG4'},
    
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-MRL1'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-MRL2'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-MRL3'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-MRL4'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-BTM1'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-BTM2'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-BTM3'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-BTM4'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-DST_'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-HOM_'},
    {'OUTPUT_DIR': '20250507-MCU-Net-Exclude-RIC_'},
]
# Loop through parameter sets and run the notebook
for params in parameter_sets:
    # Create a temporary file for the output notebook
    with tempfile.NamedTemporaryFile(suffix=".ipynb") as temp_output:
        pm.execute_notebook(
            'Main_without_Annealing.ipynb',  # Path to your input notebook
            temp_output.name,  # Path to the temporary output notebook
            parameters=params
        )