#!/usr/bin/python
#
# This script collects statistics on the image markup data sets
#

import sys
import os
import string
import multiprocessing
from math import sqrt

# parameters
input_dir = "/home/kaibo/projects/gpu-db/emory/dataset"
output_dir = "/home/kaibo/projects/gpu-db/emory/dataset3"

#
# this function parses a polygon string, returning characteristics of this polygon
# returned array is in the form: [mbr_size, mbr_wh_ratio, nr_vertices]
#
def parse_poly(line, xOffset, yOffset):
	terms = line.strip()[:-1].split(',"');
	leftDownX = sys.maxint;
	leftDownY = sys.maxint;
	rightUpX = 0;
	rightUpY = 0;
	tmp = [];
	for i in terms[1].split(', '):
		x, y = i.split();
		x = int(x);
		y = int(y);
		if (x < leftDownX):
			leftDownX = x;
		if (x > rightUpX):
			rightUpX = x;
		if (y < leftDownY):
			leftDownY = y;
		if (y > rightUpY):
			rightUpY = y;
		tmp.append((str(int(x)-xOffset), str(int(y)-yOffset)));
	size = str(len(tmp));
	leftDownX = str(leftDownX-xOffset);
	leftDownY = str(leftDownY-yOffset);
	rightUpX = str(rightUpX-xOffset);
	rightUpY = str(rightUpY-yOffset);
	thisLine = '0'*(4-len(size))+ size + ', ' + '0'*(4-len(leftDownX))+ leftDownX + ' ' +  \
		'0'*(4-len(rightUpX))+ rightUpX + ' ' + '0'*(4-len(leftDownY))+ leftDownY + ' ' +  \
		'0'*(4-len(rightUpY))+ rightUpY + ',';
	for (x, y) in tmp: # the last point is the first point, so we only need tmp[:-1]
		thisLine += ' ' + '0'*(4-len(x))+ x + ' ' + '0'*(4-len(y))+y + ',';
	thisLine += '\n';	
	return thisLine, len(tmp);


def parse_tile(tile_path, out_path, tileName):
	terms = tileName.split('-')
	xOffset = int(terms[1])
	yOffset = int(terms[2])
	fin = open(tile_path, "r")
	lines = fin.readlines()
	fin.close()

	outbuf = []
	nr_polys = 0
	nr_vertices = 0
	
	for line in lines:
		
		thisLine, nv = parse_poly(line, xOffset, yOffset)
		outbuf.append(thisLine);
		# update stats
		nr_polys += 1
		nr_vertices += nv

	fout = open(out_path, "w")
	fout.write(str(nr_polys) + ", " + str(nr_vertices) + "\n")
	for iout in outbuf:
		fout.write(iout)
	fout.close()
	
	return 0


def parse_image(img_dir, out_dir):
	tiles = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]

	# spawn a swamp of processes to prase tiles in parallel
	pool  = multiprocessing.Pool(8)
	results = \
	[pool.apply_async(parse_tile, [os.path.join(img_dir, tile), os.path.join(out_dir, tile), tile]) for tile in tiles]
	
	# collect statistics
	for itile in range(0, len(tiles)):
		# sync with a dummy return value
		ret_dummy = results[itile].get()


# main
datasets = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]
# parse each dataset (1 and 2)
for ds in datasets:
	ds_dir = os.path.join(input_dir, ds)
	out_dir = os.path.join(output_dir, ds)
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	images = [f for f in os.listdir(ds_dir) if os.path.isdir(os.path.join(ds_dir, f))]
	# parse each image in the dataset (18 images in each dataset)
	for img in images:
		# make output dirs
		tmp_dir = os.path.join(out_dir, img)
		if not os.path.exists(tmp_dir):
			os.makedirs(tmp_dir)
		tmp_dir = os.path.join(tmp_dir, "markup")
		if not os.path.exists(tmp_dir):
			os.makedirs(tmp_dir)
		# begin parsing
		parse_image(os.path.join(ds_dir, img + "/markup"), tmp_dir)


