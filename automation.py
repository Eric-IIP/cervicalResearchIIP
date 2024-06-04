
import papermill as pm
import tempfile
# Define your parameters



    
    # # Add more parameter sets as needed


parameter_sets = [
    {'N_BLOCK': 4, 'LR': 0.01,  'OUTPUT_DIR': 'o5'},
    {'N_BLOCK': 4, 'LR': 0.001, 'OUTPUT_DIR': 'o6'},
    {'N_BLOCK': 5, 'LR': 0.01,  'OUTPUT_DIR': 'o7'},
    {'N_BLOCK': 5, 'LR': 0.001, 'OUTPUT_DIR': 'o8'},
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
            'mainUNet.ipynb',  # Path to your input notebook
            temp_output.name,  # Path to the temporary output notebook
            parameters=params
        )