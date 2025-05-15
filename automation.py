
import papermill as pm
import tempfile
import random
import random
# Define your parameters

random_seeds = []

random_seeds.append(42)

for i in range(99):
    random_seeds.append(random.randint(1,200000))

parameter_sets = []
channels = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 108]

for channel in channels:
    for i in range(100):    
        trial = {'OUTPUT_DIR': '20250511-' + str(channel) + 'xChannels-Seed' + str(random_seeds[i]), 'IN_CHANNEL': channel, 'RANDOM_SEED': random_seeds[i]}
        parameter_sets.append(trial)
    

# parameter_sets = [
#     {'OUTPUT_DIR': '20250511-10xChannels-Seed42', 'IN_CHANNEL': 10, 'RANDOM_SEED': 42},
# ]
# Loop through parameter sets and run the notebook


for params in parameter_sets:
    # Create a temporary file for the output notebook
    with tempfile.NamedTemporaryFile(suffix=".ipynb") as temp_output:
        pm.execute_notebook(
            'Main_without_Annealing.ipynb',  # Path to your input notebook
            temp_output.name,  # Path to the temporary output notebook
            parameters=params
        )