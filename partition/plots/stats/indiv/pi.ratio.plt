#!/usr/bin/gnuplot

reset

# eps
#set terminal postscript eps size 3.5,2.62 enhanced color font 'Helvetica,20'
set terminal postscript eps size 3.5,2.62 enhanced color font ',28'
set output 'ratiopais.eps'

# latex
#set terminal epslatex size 3.5,2.62 # color colortext
#set output 'ratiopais.eps'

# color style 
load '../../gnucolor.plt'

# Legend
# set key at 6.1,1.3
set key autotitle columnheader
set key right top
set key box

# Axes label 
set xlabel 'Bucket size' offset 0,0.5 
#set ylabel 'Time (min)'

# Axes scale 
set logscale x 10
set logscale y 10
set format x "10^{%L}"
set format y "10^{%L}"

# Axes ranges
set xrange[10:200000]
# set yrange[0:2000]

# Axis tics 
# set xtics font "Times-Roman, 30"
# set xtics (20, 50, 80, 120, 160, 200)
# set xtics ('-2{/Symbol p}' -2*pi, '-{/Symbol p}' -pi, 0, '{/Symbol p}' pi, '2{/Symbol p}' 2*pi)
# set ytics 1
# set tics scale 0.75

unset key
# Plot

plot 'pi.ratio.dat' using 1:2  notitle with lines ls 3,	\
     '' using 1:2  with points ls 3, \
     '' using 1:3  notitle with lines ls 2, \
     '' using 1:3  with points ls 2, \
     '' using 1:4  notitle with lines ls 1, \
     '' using 1:4  with points ls 1, \
     '' using 1:5  notitle with lines ls 4, \
     '' using 1:5  with points ls 4, \
     '' using 1:6  notitle with lines ls 6, \
     '' using 1:6  with points ls 6, \
     '' using 1:7  notitle with lines ls 5, \
     '' using 1:7  with points ls 5
