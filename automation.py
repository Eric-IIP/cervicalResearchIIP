
import papermill as pm
import tempfile
import random
# Define your parameters

parameter_sets = [
    
    # {'N_BLOCK': 4, 'LR': 0.01, 'IN_CHANNEL': 45,  'OUTPUT_DIR': '25', 'INPUT': ['N1', 'N2'], 'ANNEALING': [], 'VALIDATION': ['N3'], 'TEST': ['oN4']},
    # {'N_BLOCK': 4, 'LR': 0.01, 'IN_CHANNEL': 45,  'OUTPUT_DIR': '26', 'INPUT': ['N1', 'N3'], 'ANNEALING': [], 'VALIDATION': ['N2'], 'TEST': ['oN4']},
    # {'N_BLOCK': 4, 'LR': 0.01, 'IN_CHANNEL': 45,  'OUTPUT_DIR': '27', 'INPUT': ['N2', 'N3'], 'ANNEALING': [], 'VALIDATION': ['N1'], 'TEST': ['oN4']},
    
    # {'N_BLOCK': 4, 'LR': 0.01, 'IN_CHANNEL': 45,  'OUTPUT_DIR': '28', 'INPUT': ['N1', 'N2'], 'ANNEALING': [], 'VALIDATION': ['N4'], 'TEST': ['oN3']},
    # {'N_BLOCK': 4, 'LR': 0.01, 'IN_CHANNEL': 45,  'OUTPUT_DIR': '29', 'INPUT': ['N1', 'N4'], 'ANNEALING': [], 'VALIDATION': ['N2'], 'TEST': ['oN3']},
    # {'N_BLOCK': 4, 'LR': 0.01, 'IN_CHANNEL': 45,  'OUTPUT_DIR': '30', 'INPUT': ['N2', 'N4'], 'ANNEALING': [], 'VALIDATION': ['N1'], 'TEST': ['oN3']},
    
    # {'N_BLOCK': 4, 'LR': 0.01, 'IN_CHANNEL': 45,  'OUTPUT_DIR': '31', 'INPUT': ['N1', 'N3'], 'ANNEALING': [], 'VALIDATION': ['N4'], 'TEST': ['oN2']},
    # {'N_BLOCK': 4, 'LR': 0.01, 'IN_CHANNEL': 45,  'OUTPUT_DIR': '32', 'INPUT': ['N1', 'N4'], 'ANNEALING': [], 'VALIDATION': ['N3'], 'TEST': ['oN2']},
    # {'N_BLOCK': 4, 'LR': 0.01, 'IN_CHANNEL': 45,  'OUTPUT_DIR': '33', 'INPUT': ['N3', 'N4'], 'ANNEALING': [], 'VALIDATION': ['N1'], 'TEST': ['oN2']},
    
    # {'N_BLOCK': 4, 'LR': 0.01, 'IN_CHANNEL': 45,  'OUTPUT_DIR': '34', 'INPUT': ['N2', 'N3'], 'ANNEALING': [], 'VALIDATION': ['N4'], 'TEST': ['oN1']},
    # {'N_BLOCK': 4, 'LR': 0.01, 'IN_CHANNEL': 45,  'OUTPUT_DIR': '35', 'INPUT': ['N2', 'N4'], 'ANNEALING': [], 'VALIDATION': ['N3'], 'TEST': ['oN1']},
    #{'N_BLOCK': 4, 'LR': 0.01, 'IN_CHANNEL': 45,  'OUTPUT_DIR': '36', 'INPUT': ['N3', 'N4'], 'ANNEALING': [], 'VALIDATION': ['N2'], 'TEST': ['oN1']},
    
    
    ### crf enhance
    {'EXPERIMENT_NUMBER': 13},
    {'EXPERIMENT_NUMBER': 14},
    {'EXPERIMENT_NUMBER': 15},
    
    {'EXPERIMENT_NUMBER': 16},
    {'EXPERIMENT_NUMBER': 17},
    {'EXPERIMENT_NUMBER': 18},
    
    {'EXPERIMENT_NUMBER': 19},
    {'EXPERIMENT_NUMBER': 20},
    {'EXPERIMENT_NUMBER': 21},
    
    {'EXPERIMENT_NUMBER': 22},
    {'EXPERIMENT_NUMBER': 23},
    {'EXPERIMENT_NUMBER': 24},
    
    
    
    
    # Add more parameter sets as needed
]
# Loop through parameter sets and run the notebook
for params in parameter_sets:
    # Create a temporary file for the output notebook
    with tempfile.NamedTemporaryFile(suffix=".ipynb") as temp_output:
        pm.execute_notebook(
            'crfEnhance.ipynb',  # Path to your input notebook
            temp_output.name,  # Path to the temporary output notebook
            parameters=params
        )