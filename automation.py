
import papermill as pm
import tempfile
import random
# Define your parameters



    
    # # Add more parameter sets as needed

# N1-1 -> N6
# N1-2 -> N7
# N1-3 -> N8
# N1-4 -> N9
# N1-5 -> N10
# N1-6 -> N11
# N1-7 -> N12
# N1-8 -> N13
# N1-9 -> N14
# N1-10 -> N15

# N2-1 -> N16
# N2-2 -> N17
# N2-3 -> N18
# N2-4 -> N19
# N2-5 -> N20
# N2-6 -> N21
# N2-7 -> N22
# N2-8 -> N23
# N2-9 -> N24
# N2-10 -> N25

# N3-1 -> N26
# N3-2 -> N27
# N3-3 -> N28
# N3-4 -> N29
# N3-5 -> N30
# N3-6 -> N31
# N3-7 -> N32
# N3-8 -> N33
# N3-9 -> N34
# N3-10 -> N35

# N4-1 -> N36
# N4-2 -> N37
# N4-3 -> N38
# N4-4 -> N39
# N4-5 -> N40
# N4-6 -> N41
# N4-7 -> N42
# N4-8 -> N43
# N4-9 -> N44

parameter_sets = [
    # 1-12 UNet#
    # 13-24 UNet
    # N1,2 and half of N3 training, N3 half val, N4 test
    # {'N_BLOCK': 4, 'LR': 0.01, 'IN_CHANNEL': 45,  'OUTPUT_DIR': '13', 'INPUT': ['N1', 'N2','N3-1-', 'N3-2-', 'N3-3-', 'N3-4-', 'N3-5-'], 'ANNEALING': [], 'VALIDATION': ['N3-6-', 'N3-7-', 'N3-8-', 'N3-9-', 'N3-10-'], 'TEST': ['oN4']},
    # {'N_BLOCK': 4, 'LR': 0.01, 'IN_CHANNEL': 45,  'OUTPUT_DIR': '14', 'INPUT': ['N1', 'N3', 'N2-1-', 'N2-2-', 'N2-3-', 'N2-4-', 'N2-5-'], 'ANNEALING': [], 'VALIDATION': ['N2-6-', 'N2-7-', 'N2-8-', 'N2-9-', 'N2-10-'], 'TEST': ['oN4']},
    # {'N_BLOCK': 4, 'LR': 0.01, 'IN_CHANNEL': 45,  'OUTPUT_DIR': '15', 'INPUT': ['N2', 'N3', 'N1-1-', 'N1-2-', 'N1-3-', 'N1-4-', 'N1-5-'], 'ANNEALING': [], 'VALIDATION': ['N1-6-', 'N1-7-', 'N1-8-', 'N1-9-', 'N1-10-'], 'TEST': ['oN4']},
    
    # {'N_BLOCK': 4, 'LR': 0.01, 'IN_CHANNEL': 45,  'OUTPUT_DIR': '16', 'INPUT': ['N1', 'N2', 'N4-1-', 'N4-2-', 'N4-3-', 'N4-4-', 'N4-5-'], 'ANNEALING': [], 'VALIDATION': ['N4-6-', 'N4-7-', 'N4-8-', 'N4-9-'], 'TEST': ['oN3']},
    # {'N_BLOCK': 4, 'LR': 0.01, 'IN_CHANNEL': 45,  'OUTPUT_DIR': '17', 'INPUT': ['N1', 'N4', 'N2-1-', 'N2-2-', 'N2-3-', 'N2-4-', 'N2-5-'], 'ANNEALING': [], 'VALIDATION': ['N2-6-', 'N2-7-', 'N2-8-', 'N2-9-', 'N2-10-'], 'TEST': ['oN3']},
    # {'N_BLOCK': 4, 'LR': 0.01, 'IN_CHANNEL': 45,  'OUTPUT_DIR': '18', 'INPUT': ['N2', 'N4', 'N1-1-', 'N1-2-', 'N1-3-', 'N1-4-', 'N1-5-'], 'ANNEALING': [], 'VALIDATION': ['N1-6-', 'N1-7-', 'N1-8-', 'N1-9-', 'N1-10-'], 'TEST': ['oN3']},
    
    # {'N_BLOCK': 4, 'LR': 0.01, 'IN_CHANNEL': 45,  'OUTPUT_DIR': '19', 'INPUT': ['N1', 'N3', 'N4-1-', 'N4-2-', 'N4-3-', 'N4-4-', 'N4-5-'], 'ANNEALING': [], 'VALIDATION': ['N4-6-', 'N4-7-', 'N4-8-', 'N4-9-'], 'TEST': ['oN2']},
    # {'N_BLOCK': 4, 'LR': 0.01, 'IN_CHANNEL': 45,  'OUTPUT_DIR': '20', 'INPUT': ['N1', 'N4', 'N3-1-', 'N3-2-', 'N3-3-', 'N3-4-', 'N3-5-'], 'ANNEALING': [], 'VALIDATION': ['N3-6-', 'N3-7-', 'N3-8-', 'N3-9-', 'N3-10-'], 'TEST': ['oN2']},
    # {'N_BLOCK': 4, 'LR': 0.01, 'IN_CHANNEL': 45,  'OUTPUT_DIR': '21', 'INPUT': ['N3', 'N4', 'N1-1-', 'N1-2-', 'N1-3-', 'N1-4-', 'N1-5-'], 'ANNEALING': [], 'VALIDATION': ['N1-6-', 'N1-7-', 'N1-8-', 'N1-9-', 'N1-10-'], 'TEST': ['oN2']},
    
    # {'N_BLOCK': 4, 'LR': 0.01, 'IN_CHANNEL': 45,  'OUTPUT_DIR': '22', 'INPUT': ['N2', 'N3', 'N4-1-', 'N4-2-', 'N4-3-', 'N4-4-', 'N4-5-'], 'ANNEALING': [], 'VALIDATION': ['N4-6-', 'N4-7-', 'N4-8-', 'N4-9-'], 'TEST': ['oN1']},
    # {'N_BLOCK': 4, 'LR': 0.01, 'IN_CHANNEL': 45,  'OUTPUT_DIR': '23', 'INPUT': ['N2', 'N4', 'N3-1-', 'N3-2-', 'N3-3-', 'N3-4-', 'N3-5-'], 'ANNEALING': [], 'VALIDATION': ['N3-6-', 'N3-7-', 'N3-8-', 'N3-9-', 'N3-10-'], 'TEST': ['oN1']},
    # {'N_BLOCK': 4, 'LR': 0.01, 'IN_CHANNEL': 45,  'OUTPUT_DIR': '24', 'INPUT': ['N3', 'N4', 'N2-1-', 'N2-2-', 'N2-3-', 'N2-4-', 'N2-5-'], 'ANNEALING': [], 'VALIDATION': ['N2-6-', 'N2-7-', 'N2-8-', 'N2-9-', 'N2-10-'], 'TEST': ['oN1']},
    
    
    
    {'N_BLOCK': 4, 'LR': 0.01, 'IN_CHANNEL': 108,  'OUTPUT_DIR': '1', 'INPUT': ['N1', 'N2'], 'ANNEALING': [], 'VALIDATION': ['N3'], 'TEST': ['oN4']},
    {'N_BLOCK': 4, 'LR': 0.01, 'IN_CHANNEL': 108,  'OUTPUT_DIR': '2', 'INPUT': ['N1', 'N3'], 'ANNEALING': [], 'VALIDATION': ['N2'], 'TEST': ['oN4']},
    {'N_BLOCK': 4, 'LR': 0.01, 'IN_CHANNEL': 108,  'OUTPUT_DIR': '3', 'INPUT': ['N2', 'N3'], 'ANNEALING': [], 'VALIDATION': ['N1'], 'TEST': ['oN4']},
    
    {'N_BLOCK': 4, 'LR': 0.01, 'IN_CHANNEL': 108,  'OUTPUT_DIR': '4', 'INPUT': ['N1', 'N2'], 'ANNEALING': [], 'VALIDATION': ['N4'], 'TEST': ['oN3']},
    {'N_BLOCK': 4, 'LR': 0.01, 'IN_CHANNEL': 108,  'OUTPUT_DIR': '5', 'INPUT': ['N1', 'N4'], 'ANNEALING': [], 'VALIDATION': ['N2'], 'TEST': ['oN3']},
    {'N_BLOCK': 4, 'LR': 0.01, 'IN_CHANNEL': 108,  'OUTPUT_DIR': '6', 'INPUT': ['N2', 'N4'], 'ANNEALING': [], 'VALIDATION': ['N1'], 'TEST': ['oN3']},
    
    {'N_BLOCK': 4, 'LR': 0.01, 'IN_CHANNEL': 108,  'OUTPUT_DIR': '7', 'INPUT': ['N1', 'N3'], 'ANNEALING': [], 'VALIDATION': ['N4'], 'TEST': ['oN2']},
    {'N_BLOCK': 4, 'LR': 0.01, 'IN_CHANNEL': 108,  'OUTPUT_DIR': '8', 'INPUT': ['N1', 'N4'], 'ANNEALING': [], 'VALIDATION': ['N3'], 'TEST': ['oN2']},
    {'N_BLOCK': 4, 'LR': 0.01, 'IN_CHANNEL': 108,  'OUTPUT_DIR': '9', 'INPUT': ['N3', 'N4'], 'ANNEALING': [], 'VALIDATION': ['N1'], 'TEST': ['oN2']},
    
    {'N_BLOCK': 4, 'LR': 0.01, 'IN_CHANNEL': 108,  'OUTPUT_DIR': '10', 'INPUT': ['N2', 'N3'], 'ANNEALING': [], 'VALIDATION': ['N4'], 'TEST': ['oN1']},
    {'N_BLOCK': 4, 'LR': 0.01, 'IN_CHANNEL': 108,  'OUTPUT_DIR': '11', 'INPUT': ['N2', 'N4'], 'ANNEALING': [], 'VALIDATION': ['N3'], 'TEST': ['oN1']},
    {'N_BLOCK': 4, 'LR': 0.01, 'IN_CHANNEL': 108,  'OUTPUT_DIR': '12', 'INPUT': ['N3', 'N4'], 'ANNEALING': [], 'VALIDATION': ['N2'], 'TEST': ['oN1']},
    
    
    # Add more parameter sets as needed
]

# Loop through parameter sets and run the notebook
# for params in parameter_sets:
#     pm.execute_notebook(
#         'test.ipynb',  # Path to your input notebook
#         f'test_output_notebook_{params["OUTPUT_DIR"]}.ipynb',  # Path to save the executed notebook
#         parameters=params
#     )
# Loop through parameter sets and run the notebook
for params in parameter_sets:
    # Create a temporary file for the output notebook
    with tempfile.NamedTemporaryFile(suffix=".ipynb") as temp_output:
        pm.execute_notebook(
            'Main_without_Annealing.ipynb',  # Path to your input notebook
            temp_output.name,  # Path to the temporary output notebook
            parameters=params
        )
        
        
        
        
        
        
        
        
        
        
        
## old cross val format
# {'N_BLOCK': 4, 'LR': 0.01, 'IN_CHANNEL': 45,  'OUTPUT_DIR': '1-2', 'INPUT': 'N1-', 'ANNEALING': 'N3-', 'VALIDATION': 'N2-', 'TEST': 'N4-'},
#     {'N_BLOCK': 4, 'LR': 0.01, 'IN_CHANNEL': 45,  'OUTPUT_DIR': '1-3', 'INPUT': 'N2-', 'ANNEALING': 'N3-', 'VALIDATION': 'N1-', 'TEST': 'N4-'},
    
#     {'N_BLOCK': 4, 'LR': 0.01, 'IN_CHANNEL': 45,  'OUTPUT_DIR': '2-1', 'INPUT': 'N1-', 'ANNEALING': 'N2-', 'VALIDATION': 'N4-', 'TEST': 'N3-'},
#     {'N_BLOCK': 4, 'LR': 0.01, 'IN_CHANNEL': 45,  'OUTPUT_DIR': '2-2', 'INPUT': 'N1-', 'ANNEALING': 'N4-', 'VALIDATION': 'N2-', 'TEST': 'N3-'},
#     {'N_BLOCK': 4, 'LR': 0.01, 'IN_CHANNEL': 45,  'OUTPUT_DIR': '2-3', 'INPUT': 'N2-', 'ANNEALING': 'N4-', 'VALIDATION': 'N1-', 'TEST': 'N3-'},

#     {'N_BLOCK': 4, 'LR': 0.01, 'IN_CHANNEL': 45,  'OUTPUT_DIR': '3-1', 'INPUT': 'N1-', 'ANNEALING': 'N3-', 'VALIDATION': 'N4-', 'TEST': 'N2-'},
#     {'N_BLOCK': 4, 'LR': 0.01, 'IN_CHANNEL': 45,  'OUTPUT_DIR': '3-2', 'INPUT': 'N1-', 'ANNEALING': 'N4-', 'VALIDATION': 'N3-', 'TEST': 'N2-'},
#     {'N_BLOCK': 4, 'LR': 0.01, 'IN_CHANNEL': 45,  'OUTPUT_DIR': '3-3', 'INPUT': 'N3-', 'ANNEALING': 'N4-', 'VALIDATION': 'N1-', 'TEST': 'N2-'},

#     {'N_BLOCK': 4, 'LR': 0.01, 'IN_CHANNEL': 45,  'OUTPUT_DIR': '4-1', 'INPUT': 'N2-', 'ANNEALING': 'N3-', 'VALIDATION': 'N4-', 'TEST': 'N1-'},
#     {'N_BLOCK': 4, 'LR': 0.01, 'IN_CHANNEL': 45,  'OUTPUT_DIR': '4-2', 'INPUT': 'N2-', 'ANNEALING': 'N4-', 'VALIDATION': 'N3-', 'TEST': 'N1-'},
#     {'N_BLOCK': 4, 'LR': 0.01, 'IN_CHANNEL': 45,  'OUTPUT_DIR': '4-3', 'INPUT': 'N3-', 'ANNEALING': 'N4-', 'VALIDATION': 'N2-', 'TEST': 'N1-'},