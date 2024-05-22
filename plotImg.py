import matplotlib.pyplot as plt
import numpy as np

def plot_images(images, titles=None):
    num_images = len(images)
    num_cols = 3  # 3 columns
    num_rows = (num_images - 1) // num_cols + 1

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(0.5, 0.5))
    axes = axes.flatten()

    for i, (image, ax) in enumerate(zip(images, axes)):
        ax.imshow(image)
        ax.axis('off')
        if titles is not None:
            ax.set_title(titles[i])

    # Hide any remaining empty subplots
    for ax in axes[num_images:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# Example usage:
# Assuming 'images' is a list containing your NumPy arrays
# And 'titles' is a list of titles for each image (optional)
images = [np.random.random((100, 100)) for _ in range(9)]  # Example images
titles = ['Image {}'.format(i+1) for i in range(len(images))]  # Example titles
plot_images(images, titles)