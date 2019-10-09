import glob
import os
img_path=glob.glob("DUTS-TR/DUTS-TR-Image/*.jpg")
print(len(img_path))
mask_path=glob.glob("DUTS-TR/DUTS-TR-Mask/*.png")
print(len(mask_path))