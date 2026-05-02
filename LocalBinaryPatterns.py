import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
from skimage import data


def lbp_single_pixel(image, x, y):
    center = image[x, y]
    pattern = []
    

    neighbors = [(x-1, y-1), (x-1, y), (x-1, y+1),
                 (x, y+1),   (x+1, y+1), (x+1, y),
                 (x+1, y-1), (x, y-1)]
    
    for nx, ny in neighbors:
        if image[nx, ny] >= center:
            pattern.append(1)
        else:
            pattern.append(0)

    power = 7
    decimal_val = 0
    for val in pattern:
        decimal_val += val * (2**power)
        power -= 1
        
    return decimal_val, pattern

padded_image = np.array([
    [ 0,  0,  0,  0,  0],
    [ 0, 10, 20, 30,  0],
    [ 0,  5, 15, 25,  0],
    [ 0, 40, 50, 60,  0],
    [ 0,  0,  0,  0,  0]
])
center_x, center_y = 2, 2

lbp_val, binary_pattern = lbp_single_pixel(padded_image, center_x, center_y)
print(f"From Scratch LBP for center (15): {lbp_val}")
print(f"Binary pattern: {binary_pattern}")



img = data.camera()


radius = 1 # radius for neighbors
n_points = 8 * radius 
METHOD = 'uniform' 


lbp = local_binary_pattern(img, n_points, radius, METHOD)



n_bins = int(lbp.max() + 1)
hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))

hist = hist.astype("float")
hist /= (hist.sum() + 1e-7)


plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(padded_image, cmap='gray', interpolation='nearest')
plt.title('Problem 3x3 Grid (padded)')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(lbp, cmap='gray')
plt.title('LBP Map (from skimage)')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.bar(np.arange(n_bins), hist)
plt.title('LBP Histogram (skimage uniform method)')
plt.xlabel('LBP Code')
plt.ylabel('Frequency (Normalized)')

plt.tight_layout()
plt.show()
