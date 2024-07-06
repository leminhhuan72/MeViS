from PIL import Image
import numpy as np
import cv2
import sys
np.set_printoptions(threshold=sys.maxsize)


mask_path = '00000.png'
mask = Image.open(mask_path).convert('P')
obj_id = 1
mask = np.array(mask)
mask = (mask==obj_id).astype(np.int32)

mask_image = Image.fromarray((mask * 255).astype(np.uint8))

# Save the mask image
mask_image.save('saved_mask.png')
