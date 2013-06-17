/* XXL: The eXtensible and fleXible Library for data processing

Copyright (C) 2000-2013 Prof. Dr. Bernhard Seeger
                        Head of the Database Research Group
                        Department of Mathematics and Computer Science
                        University of Marburg
                        Germany

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 3 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library;  If not, see <http://www.gnu.org/licenses/>. 

    http://code.google.com/p/xxl/

 */
package xxl.core.spatial;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;

import xxl.core.cursors.Cursor;
import xxl.core.cursors.sources.CollectionCursor;
import xxl.core.cursors.sources.io.FileInputCursor;
import xxl.core.io.converters.ConvertableConverter;
import xxl.core.spatial.histograms.MHistogram;
import xxl.core.spatial.histograms.RGOhist;
import xxl.core.spatial.rectangles.DoublePointRectangle;


public class TestH1 {
	
	public static double sampleRate = 0.9;
	public static double alpha = 0.1; // under sampling ratio 
	public static double rTreeRatio= 0.4;
	public static int gridSize= 9;
	
	
	public static PrintStream getPrintStream(String output) throws IOException{
		return new PrintStream(new File(output)); 
	}
	
	
	public static Cursor<DoublePointRectangle> getData(String path) throws IOException {
		Cursor<DoublePointRectangle> data ;
		List<DoublePointRectangle> rectangles ;
		
		if (path.endsWith("rec"))
		{
			data = new FileInputCursor<DoublePointRectangle>(
					new ConvertableConverter<DoublePointRectangle>(RGOhist.factoryFunction(2)), 
					new File(path));
		}
		else {
			BufferedReader br = new BufferedReader(new FileReader (path));
			String line ;
			rectangles = new ArrayList<DoublePointRectangle>();
			
			while (null != (line = br.readLine()))
			{
				double [] leftCorner = new double [2];
				double [] rightCorner = new double [2];
				String [] sp = line.split("\\s+");
				
				leftCorner[0] = Double.parseDouble(sp[1]);
				leftCorner[1] = Double.parseDouble(sp[2]);
				rightCorner[0] = Double.parseDouble(sp[3]);
				rightCorner[1] = Double.parseDouble(sp[4]);
				
				rectangles.add(new DoublePointRectangle(leftCorner, rightCorner));
			}
			
			br.close();
			System.err.println("Collection size: " + rectangles.size());
			data = new CollectionCursor<DoublePointRectangle>(rectangles);
		}
		
		return data ;
	}
	

	public static void main(String[] args) throws IOException {
		if (args.length <8 )
		{
			System.err.println("Usage: "
					+ TestH1.class.getSimpleName() 
					+" [number of buckets] [input Data] [output File] " 
					+" [temp_path = [temoprary path] "
					+" [minskew grid size = 10] "
					+" [minskew grid size = 10] "
					+" [minskew grid size = 10] "
					+" []");
			System.exit(0);
		}

		int numberOfBuckets = Integer.parseInt(args[0]) ;
		String inPath = args[1]; // data path
		String outPath = args[2]; // data path
		String tempPath =  args[3];
		gridSize = Integer.parseInt(args[4]);
		alpha = Double.parseDouble(args[5]);
		rTreeRatio = Double.parseDouble(args[6]);
		sampleRate = Double.parseDouble(args[7]);
		
		
		System.err.println("++++++++++++++++++++++++++++++++++++\n");
		System.err.println("Data: " + inPath);
		
		HistogramEval eval = new HistogramEval(getData(inPath),tempPath);
		MHistogram histogram  = null;
		
		System.err.println("Buckets " + numberOfBuckets);
		
		// rtree loaded bulk loaded using hilbert curve equi sized partitioning
		System.err.println("Building RTree histogram..");
		eval.buildRTreeHist(numberOfBuckets, true);
		histogram = eval.getRTreeHist();
		eval.dumpHistogram(histogram,getPrintStream(outPath+".rtree.txt"));
		
		// minskew 
		System.err.println("Building minSkew histogram..");
		eval.buildMinSkewHist(numberOfBuckets, gridSize, true);														// 2^7 grid
		histogram = eval.getMinSkewHist();
		eval.dumpHistogram(histogram,getPrintStream(outPath+".minskew.txt"));
		
		// standard min skew 2^7 x 2^7 grid and three refinerment steps
		System.err.println("Building Progressive minSkew histogram..");
		eval.buildMinSkewProgressiveHist(numberOfBuckets, gridSize, 3, true);
		histogram = eval.getMinSkewProgressiveRefinementHistogram();
		eval.dumpHistogram(histogram,getPrintStream(outPath+".minskewrefine.txt"));
		
		// RV histogram
		System.err.println("Building RV histogram..");
		eval.buildRHistogramV(numberOfBuckets, rTreeRatio, true); 
		histogram = eval.getRhistogram_V();
		eval.dumpHistogram(histogram,getPrintStream(outPath+".rv.txt"));
		
		//rkHist
		System.err.println("Building rkHist histogram..");
		eval.buildRKHist(numberOfBuckets, alpha, HistogramEval.BLOCKSIZE, true); // rkHist																		// Method
		histogram = eval.getRkHist();
		eval.dumpHistogram(histogram,getPrintStream(outPath+".rkhist.txt"));

		//STForest 
		/* System.err.println("Building STForest histogram..");
		eval.buildSTForestHist(numberOfBuckets, sampleRate, true);
		histogram = eval.getSTHistForest();
		eval.dumpHistogram(histogram,getPrintStream(outPath+".stforest.txt"));
		
		System.err.println("Building soptRTree histogram..");
		eval.buildSOPTRtreeHist(numberOfBuckets, true);
		histogram = eval.getSoptHist();
		eval.dumpHistogram(histogram,getPrintStream(outPath+".soptrtree.txt"));
		*/

		System.err.println("Done.");
		System.err.println("++++++++++++++++++++++++++++++++++++\n");
	}

}
