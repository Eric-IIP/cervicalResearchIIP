
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
    {'EXPERIMENT_NUMBER': 1},
    {'EXPERIMENT_NUMBER': 2},
    {'EXPERIMENT_NUMBER': 3},
    
    {'EXPERIMENT_NUMBER': 4},
    {'EXPERIMENT_NUMBER': 5},
    {'EXPERIMENT_NUMBER': 6},
    
    {'EXPERIMENT_NUMBER': 7},
    {'EXPERIMENT_NUMBER': 8},
    {'EXPERIMENT_NUMBER': 9},
    
    {'EXPERIMENT_NUMBER': 10},
    {'EXPERIMENT_NUMBER': 11},
    {'EXPERIMENT_NUMBER': 12},
    
    
    
    
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