#!/usr/bin/gnuplot
#
reset
#eps 
set terminal postscript eps size 3.5,2.62 enhanced color font ',20'
set output 'partperf.eps'
# load '../gnucolor.plt'

# wxt
# set terminal wxt size 350,262 enhanced font 'Verdana,10' persist
# png
#set terminal pngcairo size 350,262 enhanced font 'Verdana,10'
#set output 'statistics.png'
# svg
#set terminal svg size 350,262 fname 'Verdana, Helvetica, Arial, sans-serif' \
#fsize '10'
#set output 'statistics.svg'

# color definitions
set border linewidth 1.5
#set style line 2 lc rgb 'grey70' lt 1 lw 2

set style line 1 lc rgb '#1B9E77' lt 1 # dark teal
set style line 2 lc rgb '#D95F02' lt 1 # dark orange

# set style fill solid 1.0 border rgb 'grey30'

#set style fill pattern 
set style fill solid
# 1.0 border rgb "black"
set style histogram clustered 
set style data histograms

set key left top
#set border 3
#set yrange [0:5000]
#set y2range [0:200]

# plot 'perf.dat' u 2:xtic(1) axes x1y2 t "PI" ls 1 , "" u 3:xtic(1) axes x1y1 t "OSM" ls 2 
plot 'perf.dat' u ($2*10.03):xtic(1) t "PI" ls 1 , "" u 3:xtic(1) t "OSM" ls 2 
