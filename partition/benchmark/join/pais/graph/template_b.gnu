#!/usr/bin/gnuplot
#
# Demonstrates a simple usage of gnuplot.
#
# AUTHOR: Hagen Wierstorf 

reset

# eps
set terminal postscript eps size 3.5,2.62 enhanced color font 'Helvetica,20' lw 2
set output 'pbsm.eps'
set style data linespoints

# latex
# set terminal epslatex size 3.5,2.62 color colortext
# set output 'ht.tex'

# Line styles
# set border linewidth 1.5
# set style line 1 linecolor rgb '#0060ad' linetype 1 linewidth 2  # blue
# set style line 2 linecolor rgb '#dd181f' linetype 1 linewidth 2  # red
# set style line 1 lc rgb '#0060ad' lt 1 lw 2 pt 0 pi -1 ps 3
# set style line 2 lc rgb '#0060ad' pt 7 ps 1.5

#set style line 1 lc '#000000' lt 1 lw 1 pt 0 pi -1 ps 2 # black
#set style line 2 lc rgb '#5a00b3' pt 5 ps 1.5  # yahoo
#set style line 3 lc rgb '#ff0000' pt 7 ps 1.5 #  red
#set style line 4 lc rgb '#ff8000' pt 9 ps 1.5 # light brwon 
#set style line 5 lc rgb '#0000ff' pt 13 ps 1.5 # blue
#set style line 6 lc rgb '#00b300' pt 15 ps 1.5 # green

# Legend
# set key at 6.1,1.3
# Axes label 
set xlabel 'Reducer'
set ylabel 'Time '

# Axis ranges
set xrange[10:210]
# set yrange[0:2000]

# Axis labels
set xtics (20, 40, 60, 80, 100, 140, 190)
# set xtics ('-2{/Symbol p}' -2*pi, '-{/Symbol p}' -pi, 0, '{/Symbol p}' pi, '2{/Symbol p}' 2*pi)
# set ytics 1
# set tics scale 0.75
set key autotitle columnhead
# Plot

set datafile separator ','

plot 'pbsm.csv' using 1:2  with linespoints,	\
     '' using 1:3  with linespoints, \
     '' using 1:4  with linespoints, \
     '' using 1:5  with linespoints, \
     '' using 1:6  with linespoints, \
     '' using 1:7  with linespoints, \
     '' using 1:8  with linespoints

