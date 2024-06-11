
import papermill as pm
import tempfile
# Define your parameters



    
    # # Add more parameter sets as needed


parameter_sets = [
    # {'N_BLOCK': 5, 'LR': 0.01,  'OUTPUT_DIR': '0-a', 'INPUT': 'N1-', 'ANNEALING': 'N2-', 'VALIDATION': 'N3-', 'TEST': 'N4-'},
    # {'N_BLOCK': 5, 'LR': 0.01,  'OUTPUT_DIR': '0-b', 'INPUT': 'N1-', 'ANNEALING': 'N3-', 'VALIDATION': 'N2-', 'TEST': 'N4-'},
    # {'N_BLOCK': 5, 'LR': 0.01,  'OUTPUT_DIR': '0-c', 'INPUT': 'N2-', 'ANNEALING': 'N3-', 'VALIDATION': 'N1-', 'TEST': 'N4-'},
    
    # {'N_BLOCK': 5, 'LR': 0.01,  'OUTPUT_DIR': '1-a', 'INPUT': 'N1-', 'ANNEALING': 'N2-', 'VALIDATION': 'N4-', 'TEST': 'N3-'},
    # {'N_BLOCK': 5, 'LR': 0.01,  'OUTPUT_DIR': '1-b', 'INPUT': 'N1-', 'ANNEALING': 'N4-', 'VALIDATION': 'N2-', 'TEST': 'N3-'},
    # {'N_BLOCK': 5, 'LR': 0.01,  'OUTPUT_DIR': '1-c', 'INPUT': 'N2-', 'ANNEALING': 'N4-', 'VALIDATION': 'N1-', 'TEST': 'N3-'},

    {'N_BLOCK': 5, 'LR': 0.01,  'OUTPUT_DIR': '2-a', 'INPUT': 'N1-', 'ANNEALING': 'N3-', 'VALIDATION': 'N4-', 'TEST': 'N2-'},
    {'N_BLOCK': 5, 'LR': 0.01,  'OUTPUT_DIR': '2-b', 'INPUT': 'N1-', 'ANNEALING': 'N4-', 'VALIDATION': 'N3-', 'TEST': 'N2-'},
    {'N_BLOCK': 5, 'LR': 0.01,  'OUTPUT_DIR': '2-c', 'INPUT': 'N3-', 'ANNEALING': 'N4-', 'VALIDATION': 'N1-', 'TEST': 'N2-'},

    # {'N_BLOCK': 5, 'LR': 0.01,  'OUTPUT_DIR': '3-a', 'INPUT': 'N2-', 'ANNEALING': 'N3-', 'VALIDATION': 'N4-', 'TEST': 'N1-'},
    # {'N_BLOCK': 5, 'LR': 0.01,  'OUTPUT_DIR': '3-b', 'INPUT': 'N2-', 'ANNEALING': 'N4-', 'VALIDATION': 'N3-', 'TEST': 'N1-'},
    # {'N_BLOCK': 5, 'LR': 0.01,  'OUTPUT_DIR': '3-c', 'INPUT': 'N3-', 'ANNEALING': 'N4-', 'VALIDATION': 'N2-', 'TEST': 'N1-'},

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