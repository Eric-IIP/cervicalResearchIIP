
import papermill as pm
import tempfile
import random
# Define your parameters

parameter_sets = []

for i in range(30):
    random_int = random.randint(1, 100000)
    parameter_sets.append({'OUTPUT_DIR': f'20250701-Pilot-Study-{random_int}', 'RANDOM_SEED': random_int})
# Loop through parameter sets and run the notebook
for params in parameter_sets:
    # Create a temporary file for the output notebook
    with tempfile.NamedTemporaryFile(suffix=".ipynb") as temp_output:
        pm.execute_notebook(
            'Main_without_Annealing.ipynb',  # Path to your input notebook
            temp_output.name,  # Path to the temporary output notebook
            parameters=params
        )