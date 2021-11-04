import matplotlib.image as mpimg

# Read data into a matrix
Data = mpimg.imread('Lena.png')

# Plotting Image
import matplotlib.pyplot as plt

plt.gray()
plt.imshow(Data)

plt.show()
