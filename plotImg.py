import matplotlib.pyplot as plt
import numpy as np
import os

def plot_images(path, images, titles=None):
    num_images = len(images)
    num_cols = 3  # 3 columns
    num_rows = (num_images - 1) // num_cols + 1

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(8, 8))



    axes = axes.flatten()

    for i, (image, ax) in enumerate(zip(images, axes)):
        ax.imshow(image)
        ax.axis('off')

    # Hide any remaining empty subplots
    for ax in axes[num_images:]:
        ax.axis('off')
    plt.savefig(os.path.join(path, "output.png"))
    plt.tight_layout()
    plt.show()

# Example usage:
# Assuming 'images' is a list containing your NumPy arrays
# And 'titles' is a list of titles for each image (optional)
#images = [np.random.random((100, 100)) for _ in range(9)]  # Example images
#plot_images(images, titles)
