import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def show_two_pgm_images(image_path1, image_path2):
    # Read the PGM images using matplotlib
    img1 = mpimg.imread(image_path1)
    img2 = mpimg.imread(image_path2)

    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Display the first image on the left subplot
    axs[0].imshow(img1, cmap='gray')
    axs[0].axis('off')  # Turn off axis labels

    # Display the second image on the right subplot
    axs[1].imshow(img2, cmap='gray')
    axs[1].axis('off')  # Turn off axis labels

    plt.show()

# Example usage:
# Replace 'path/to/your/image1.pgm' and 'path/to/your/image2.pgm'
# with the paths to your two PGM image files

pgm_image_path1 = r"C:\Users\gonza\OneDrive\Escritorio\master_2023\Vision_Computer\Lab_1\p1-4students\imgs-P1\peppers.ppm"
pgm_image_path2 = r"C:\Users\gonza\OneDrive\Escritorio\master_2023\Vision_Computer\Lab_1\p1-4students\imgs-out-P1\peppers_br2.ppm"
show_two_pgm_images(pgm_image_path1, pgm_image_path2)