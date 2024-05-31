
import papermill as pm
import tempfile
# Define your parameters



    # {'N_BLOCK': 3, 'LR': 0.01,  'OUTPUT_DIR': '1'},
    # {'N_BLOCK': 3, 'LR': 0.001, 'OUTPUT_DIR': '2'},
    # {'N_BLOCK': 4, 'LR': 0.01,  'OUTPUT_DIR': '3'},
    # {'N_BLOCK': 4, 'LR': 0.001, 'OUTPUT_DIR': '4'},
    # {'N_BLOCK': 5, 'LR': 0.01,  'OUTPUT_DIR': '5'},
    # {'N_BLOCK': 5, 'LR': 0.001, 'OUTPUT_DIR': '6'},
    # # Add more parameter sets as needed


parameter_sets = [
    {'N_BLOCK': 5, 'LR': 0.01,  'OUTPUT_DIR': '5'},
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