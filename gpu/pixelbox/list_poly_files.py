#!/usr/bin/python
#
# This script collects statistics on the image markup data sets
#

import sys
import os
import string
import multiprocessing
from math import sqrt

data_root = "/home/kaibo/projects/gpu-db/emory/dataset3"
out_name = "filenames.h"
allpairs = []


fout = open(out_name, "w")

ds1_dir = os.path.join(data_root, "1")
ds2_dir = os.path.join(data_root, "2")

images = [f for f in os.listdir(ds1_dir) if os.path.isdir(os.path.join(ds1_dir, f))]
for image in images:
	img1_dir = os.path.join(ds1_dir, image)
	img1_dir = os.path.join(img1_dir, "markup")
	img2_dir = os.path.join(ds2_dir, image)
	img2_dir = os.path.join(img2_dir, "markup")

	outbuf = "char *file_names_" + image.replace(".", "_") + "[][2] = {\n"
	fout.write(outbuf)

	tiles = [f for f in os.listdir(img1_dir) if os.path.isfile(os.path.join(img1_dir, f))]
	for tile in tiles:
		tile1_path = os.path.join(img1_dir, tile)
		tile2_path = os.path.join(img2_dir, tile)

		outbuf = "\t{\"" + tile1_path + "\", \"" + tile2_path + "\"},\n"
		fout.write(outbuf);

		allpairs.append([tile1_path, tile2_path])

	outbuf = "\t{NULL, NULL}\n};\n\n"
	fout.write(outbuf)

outbuf = "char *file_names_all_images[][2] = {\n"
fout.write(outbuf)

for pair in allpairs:
	outbuf = "\t{\"" + pair[0] + "\", \"" + pair[1] + "\"},\n"
	fout.write(outbuf)
outbuf = "\t{NULL, NULL}\n};\n"
fout.write(outbuf)

fout.close()

