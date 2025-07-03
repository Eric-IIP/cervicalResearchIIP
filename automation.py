
import papermill as pm
import tempfile
import random
import json
# Define your parameters

parameter_sets = []

#random_numbers = [35468, 60131, 57903, 76888, 77846, 49697, 8526, 69188, 21199, 69675, 97682, 68421, 48592, 41825, 5270, 20289, 15303, 41559, 66189, 78723, 75327, 49600, 61658, 6436, 84854, 38853, 54877, 4704, 81736, 55931]
#random_numbers2 = [60456, 49567, 76574, 99097, 73862, 56455, 17439, 98145, 8155, 55529, 3262, 5798, 19566, 11423, 29697, 18733, 70230, 82520, 80887, 78824]

file_path = '/home/eric/Documents/cervicalResearchIIP/result_test/MCUEffectivenessReal/seeds.json'  # replace with your path
# Read list back from file
with open(file_path, 'r') as f:
    loaded_seeds = json.load(f)
    
  
print(loaded_seeds)
print(type(loaded_seeds))  


random_seeds = []
for i in range(265):
    #random_int = random.randint(1, 100000)
    #random_seeds.append(random_int)
    parameter_sets.append({'OUTPUT_DIR': f'20250701-Study-UNet-{loaded_seeds[i]}', 'RANDOM_SEED': loaded_seeds[i]})
# Loop through parameter sets and run the notebook
for params in parameter_sets:
    # Create a temporary file for the output notebook
    with tempfile.NamedTemporaryFile(suffix=".ipynb") as temp_output:
        pm.execute_notebook(
            'Main_without_Annealing.ipynb',  # Path to your input notebook
            temp_output.name,  # Path to the temporary output notebook
            parameters=params
        )
        


# # # Write list to file as JSON
# with open(file_path, 'w') as f:
#     json.dump(random_seeds, f)

# # Read list back from file
# with open(file_path, 'r') as f:
#     loaded_seeds = json.load(f)