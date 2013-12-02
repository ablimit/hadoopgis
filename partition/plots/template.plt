#!/usr/bin/gnuplot

reset

# eps
set terminal postscript eps size 3.5,2.62 enhanced color font 'Helvetica,20' lw 2
set output '_chartname_'

# latex
# set terminal epslatex size 3.5,2.62 color colortext
# set output 'ht.tex'

# Line styles
# set border linewidth 1.5
# set style line 1 linecolor rgb '#0060ad' linetype 1 linewidth 2  # blue
# set style line 2 linecolor rgb '#dd181f' linetype 1 linewidth 2  # red
# set style line 1 lc rgb '#0060ad' lt 1 lw 2 pt 0 pi -1 ps 3
# set style line 2 lc rgb '#0060ad' pt 7 ps 1.5

set style line 1 lc '#000000' lt 1 lw 1 pt 0 pi -1 ps 2 # black
set style line 2 lc rgb '#5a00b3' pt 5 ps 1.5  # yahoo
set style line 3 lc rgb '#ff0000' pt 7 ps 1.5 #  red
set style line 4 lc rgb '#ff8000' pt 9 ps 1.5 # light brwon 
set style line 5 lc rgb '#0000ff' pt 13 ps 1.5 # blue
set style line 6 lc rgb '#00b300' pt 15 ps 1.5 # green

# Legend
# set key at 6.1,1.3
# Axes label 
set xlabel 'bucket size'
# set ylabel ''
set logscale x 10
set logscale y 10
# Axis ranges
# set xrange[10:210]
# set yrange[0:2000]

# Axis labels
# set xtics (20, 50, 80, 120, 160, 200)
# set xtics ('-2{/Symbol p}' -2*pi, '-{/Symbol p}' -pi, 0, '{/Symbol p}' pi, '2{/Symbol p}' 2*pi)
# set ytics 1
# set tics scale 0.75

set key autotitle columnheader

# Plot

plot '_dataset_' using 1:2 notitle with linespoints ls 1,	\
     '' using 1:2 with points ls 2, \
     '' using 1:3 notitle with linespoints ls 1, \
     '' using 1:3 with points ls 3, \
     '' using 1:4 notitle with linespoints ls 1, \
     '' using 1:4 with points ls 4, \
     '' using 1:5 notitle with linespoints ls 1, \
     '' using 1:5 with points ls 5, \
     '' using 1:6 notitle with linespoints ls 1, \
     '' using 1:6 with points ls 6

