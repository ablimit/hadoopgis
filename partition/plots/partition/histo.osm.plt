#!/usr/bin/gnuplot
#
reset
#eps 
set terminal postscript eps size 3.5,2.62 enhanced color font ',24'
set output 'osm.perf.eps'
# load '../gnucolor.plt'

# color definitions
set border linewidth 1.5
#set style line 2 lc rgb 'grey70' lt 1 lw 2

set style line 1 lc rgb '#1B9E77' lt 1 # dark teal
set style line 2 lc rgb '#D95F02' lt 1 # dark orange

# set style fill solid 1.0 border rgb 'grey30'
# set ylabel "Time (sec) {/=15 x}10^2" enhanced
set ylabel offset 1.5 "Time (sec)" enhanced
set xtics nomirror scale 0
#set ytics nomirror

set boxwidth 0.75
set style fill solid # transparent solid 0.5 noborder

set label 1 "0.001" at graph 0.03,0.08 font ",20"
set label 2 "66.1" at graph 0.2,0.07 font ",20"
set arrow 1 from graph 0.085,0.05 to graph 0.085,0.01 head empty ls 1

plot "perf.dat" u 3:xtic(1) w boxes ls 1 notitle

# plot 'perf.dat' u 2:xtic(1) axes x1y2 t "PI" ls 1 , "" u 3:xtic(1) axes x1y1 t "OSM" ls 2 
# plot 'perf.dat' u ($2*10.03):xtic(1) t "PI" ls 1 , "" u 3:xtic(1) t "OSM" ls 2 
# plot 'perf.dat' u 2:xtic(1) ls 2 notitle 
