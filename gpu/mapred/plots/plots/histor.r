#!/usr/bin/env Rscript

library(ggplot2);
ExportPlot <- function(gplot, filename, width=3.5, height=2.62) {
    # Export plot in PDF and EPS.
    # Notice that A4: width=11.69, height=8.27
    ggsave(paste(filename, '.pdf', sep=""), gplot, width = width, height = height)
    postscript(file = paste(filename, '.eps', sep=""), horiz=FALSE, width = width, height = height)
    print(gplot)
    dev.off()
    png(file = paste(filename, '.png', sep=""), width = width * 100, height = height * 100)
    print(gplot)
    dev.off()
}

doAllPlotR <- function(filename) {
  # setwd("/Users/ablimit/Documents/proj/hadoopgis/gpu/mapred/plots/misc");
  mydata <-read.csv(paste(filename,'.csv', sep=""), header=TRUE);
  m <- ggplot(data=mydata, aes(x=factor(cpu), y=runtime, fill=gpu)) ;
  gplot = m + geom_bar(stat="identity", position=position_dodge())

    ExportPlot(gplot, filename);
}
