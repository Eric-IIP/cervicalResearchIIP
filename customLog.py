import os

def custom_logger(log_file, *args):
    directory = os.path.dirname(log_file)
    if not os.path.exists(directory):
        os.makedirs(directory)  # Create directory if it doesn't exist

    with open(log_file, "a") as f:
        if os.path.exists(log_file):
            f.write("\n")  # Start from a new line if the file exists
        for var in args:
            f.write(f"{var}: {eval(var)}\n")