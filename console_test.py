import cv2
import numpy

with open('database/metal_list', 'r+') as ml:
    metal = ml.read().splitlines()
with open('database/paper_list', 'r+') as pl:
    paper = pl.read().splitlines()
with open('database/plastic_list', 'r+') as pll:
    plastic = pll.read().splitlines()
with open('database/unknown_list', 'r+') as uf:
    U_List = uf.read().splitlines()
my_barcode = input()
print(my_barcode)
if my_barcode in metal or paper or plastic:
    myOutput = 'Recognized'
else:
    uf.write(str(my_barcode) + '\n')
    print("!!!have been done!!!")
