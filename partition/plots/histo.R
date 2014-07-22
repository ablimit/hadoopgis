#!/usr/bin/env Rscript

library(ggplot2);
ExportPlot <- function(gplot, filename, width=3.5, height=2.62) {
    # Export plot in PDF and EPS.
    # Notice that A4: width=11.69, height=8.27
    ggsave(paste(filename, '.pdf', sep=""), gplot, width = width, height = height)
    postscript(file = paste(filename, '.eps', sep=""), width = width, height = height)
    print(gplot)
    dev.off()
    png(file = paste(filename, '_.png', sep=""), width = width * 100, height = height * 100)
    print(gplot)
    dev.off()
}

setwd("/Users/ablimit/Documents/proj/papers/icde2015/misc/ggplots");
mydata <-read.csv("/Users/ablimit/Documents/proj/papers/icde2015/misc/ggplots/histo.dat");
m <- ggplot(mydata, aes(x=bs)) ;

gplot = m + geom_histogram(binwidth=20000, colour="black",fill="white") + scale_y_sqrt() +xlab("Bin Center") + ylab("Count") + annotate("text", x = 400000, y = 50000, label = "Bin width = 20K")
ExportPlot(gplot, "histoosm");
