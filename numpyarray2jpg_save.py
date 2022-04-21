# numpy to jpg
from PIL import Image
im = Image.fromarray(A)
im.save("your_file.jpg")

# jpg to numpy
import numpy as np
from PIL import Image
img = Image.open("NASA.jpg")
imgArray = np.asarray(img)
print(imgArray.shape)
