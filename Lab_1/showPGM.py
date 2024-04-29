import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def show_pgm_image(image_path):
    # Read the PGM image using matplotlib
    img = mpimg.imread(image_path)

    # Display the image using matplotlib
    plt.imshow(img, cmap='gray')
    plt.axis('off')  # Turn off axis labels
    plt.show()

# Example usage:
# Replace 'path/to/your/image.pgm' with the path to your PGM image file
pgm_image_path = r"C:\Users\gonza\OneDrive\Escritorio\master_2023\Vision_Computer\Lab_1\p1-4students\imgs-out-P1\iglesia_board.pgm"
show_pgm_image(pgm_image_path)
