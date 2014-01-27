#!/usr/bin/gnuplot

reset

# eps
set terminal postscript eps size 3.5,0.22 enhanced color font 'Times-Roman,16'
#set output 'ratioosm.eps'

# latex
#set terminal epslatex size 3.5,2.62 color colortext
set output 'statlegend.eps'

# color style 
load 'gnucolor.plt'

# Legend
set key center center horizontal samplen 2 autotitle columnheader

#unset origin
unset border
unset tics
unset label
unset arrow
unset title
unset object

set xrange [-1:2]
set yrange [-1:2]
# Plot
plot NaN t 'BSP' with points ls 1, \
     NaN t 'FG' with points ls 2, \
     NaN t 'BOS' with points ls 3, \
     NaN t 'HC' with points ls 4, \
     NaN t 'STR' with points ls 5, \
     NaN t 'SLC' with points ls 6
     #NaN t 'Strip' with points ls 7


#plot 'pltdata.dat' using 1:2  notitle with lines ls 1,	\
#     '' using 1:2  with points ls 1, \
#     '' using 1:3  notitle with lines ls 2, \
#     '' using 1:3  with points ls 2, \
#     '' using 1:4  notitle with lines ls 3, \
#     '' using 1:4  with points ls 3, \
#     '' using 1:5  notitle with lines ls 4, \
#     '' using 1:5  with points ls 4, \
#     '' using 1:6  notitle with lines ls 5, \
#     '' using 1:6  with points ls 5, \
#     '' using 1:7  notitle with lines ls 6, \
#     '' using 1:7  with points ls 6, \
#     '' using 1:8  notitle with lines ls 7, \
#     '' using 1:8  with points ls 7
