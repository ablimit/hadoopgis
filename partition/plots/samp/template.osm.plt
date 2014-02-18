#!/usr/bin/gnuplot

reset

# eps
# set terminal postscript eps size 3.5,2.62 enhanced color font 'Helvetica,20' lw 2
set terminal postscript eps size 3.5,2.62 enhanced color font ',28'
set output '_chartname_'

# latex
# set terminal epslatex size 3.5,2.62 color colortext
# set output 'ht.tex'

# color style 
load '../gnucolor.plt'

# Legend
# set key at 6.1,1.3

# Axes label 
set xlabel 'Bucket size'
# set ylabel 'Metric'

# Axes scale 
set logscale x 10
set logscale y 10
set format x "10^{%L}"
set format y "10^{%L}"

# Axes ranges
# set xrange[10:210]
# set yrange[0:2000]

# Axis tics 
# set xtics (20, 50, 80, 120, 160, 200)
# set xtics ('-2{/Symbol p}' -2*pi, '-{/Symbol p}' -pi, 0, '{/Symbol p}' pi, '2{/Symbol p}' 2*pi)
# set ytics 1
# set tics scale 0.75

set key autotitle columnheader
set key _keyposition_
# set key box
set key samplen .3

# Plot

plot '_dataset_' using 1:2  notitle with lines ls 1,	\
     '' using 1:2  t "10^{-4}" with points ls 1, \
     '' using 1:4  notitle with lines ls 2, \
     '' using 1:4  t "10^{-3}" with points ls 2, \
     '' using 1:8  notitle with lines ls 3, \
     '' using 1:8  t "10^{-2}" with points ls 3, \
     '' using 1:14  notitle with lines ls 4, \
     '' using 1:14 t "1.0" with points ls 4
     #'' using 1:6  notitle with lines ls 5, \
     #'' using 1:6  with points ls 5,\
     #'' using 1:7  notitle with lines ls 6, \
     #'' using 1:7  with points ls 6, \
     #'' using 1:8  notitle with lines ls 7, \
     #'' using 1:8  with points ls 7

