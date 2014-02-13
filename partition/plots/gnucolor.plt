# line styles for ColorBrewer Dark2
# for use with qualitative/categorical data
# provides 8 dark colors based on Set2
# compatible with gnuplot >=4.2
# author: Anna Schneider

# line styles
set style line 9 lc '#000000' lt 2 lw 1 pt 0 pi -1 ps 2 # black
set style line 1 lc rgb '#1B9E77' lt 2 pt 5 ps 2 # dark teal
set style line 2 lc rgb '#D95F02' lt 2 pt 7 ps 2 # dark orange
set style line 3 lc rgb '#7570B3' lt 2 pt 9 ps 2 # dark lilac
set style line 4 lc rgb '#E7298A' lt 2 pt 13 ps 2 # dark magenta
set style line 5 lc rgb '#66A61E' lt 2 pt 15 ps 2 # dark lime green
set style line 6 lc rgb '#E6AB02' lt 2 pt 6 ps 2 # dark banana
set style line 7 lc rgb '#A6761D' lt 2 pt 19 ps 2 # dark tan
set style line 8 lc rgb '#666666' lt 2 pt 21 ps 2 # dark gray

# palette
set palette maxcolors 9
set palette defined ( 0 '#1B9E77',\
    	    	      1 '#D95F02',\
		      2 '#7570B3',\
		      3 '#E7298A',\
		      4 '#66A61E',\
		      5 '#E6AB02',\
		      6 '#A6761D',\
		      7 '#666666',\
          8 '#000000')
