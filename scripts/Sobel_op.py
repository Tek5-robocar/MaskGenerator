from matplotlib import pyplot as plt
from skimage.data import coins
from skimage.feature import canny

edges = canny(coins/255.)

fig, ax = plt.subplots(figsize=(4, 3))

ax.imshow(edges, cmap=plt.cm.gray, interpolation='nearest')

ax.axis('off')

ax.set_title('Canny detector')

from scipy import ndimage as ndi

fill_coins = ndi.binary_fill_holes(edges)

fig, ax = plt.subplots(figsize=(4, 3))

ax.imshow(fill_coins, cmap=plt.cm.gray, interpolation='nearest')

ax.axis('off')

ax.set_title('Filling the holes')

from skimage import morphology

coins_cleaned = morphology.remove_small_objects(fill_coins, 21)

fig, ax = plt.subplots(figsize=(4, 3))

ax.imshow(coins_cleaned, cmap=plt.cm.gray, interpolation='nearest')

ax.axis('off')

ax.set_title('Removing small objects')
