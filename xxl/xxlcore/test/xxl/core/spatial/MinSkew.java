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


public class MinSkew {

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
			rectangles = new ArrayList<>();
			
			while (null != (line = br.readLine()))
			{
				double[] leftCorner = new double[2];
				double[] rightCorner = new double[2];
				String[] sp = line.split("\\s+");
				if (sp.length >= 5) {
					leftCorner[0] = Double.parseDouble(sp[1]);
					leftCorner[1] = Double.parseDouble(sp[2]);
					rightCorner[0] = Double.parseDouble(sp[3]);
					rightCorner[1] = Double.parseDouble(sp[4]);
					
					rectangles.add(new DoublePointRectangle(leftCorner,
							rightCorner));
				}
			}
			
			br.close();
			System.err.println("Collection size: " + rectangles.size());
			data = new CollectionCursor<DoublePointRectangle>(rectangles);
		}
		
		return data ;
	}
	

	public static void main(String[] args) throws IOException {
		if (args.length <5 )
		{
			System.err.println("Usage: "
					+ MinSkew.class.getSimpleName() 
					+" [number of buckets] [input Data] [output File] " 
					+" [histogram type = [ I | II ]"
					+" [X = grid size 2^(x-1)]"
					+" [show]");
			System.exit(0);
		}

		int numberOfBuckets = Integer.parseInt(args[0]) ;
		String inPath = args[1]; // data path
		String outPath = args[2]; // data path
		String tempPath =  "/tmp/hist/"; 
		String histogram_type = args[3].trim().toLowerCase();
		int gridSize = Integer.parseInt(args[4]);
		boolean showPlot = false;
		if (args.length > 5 && args[5].equalsIgnoreCase("show"))
			showPlot = true;  
		
		System.err.println("++++++++++++++++++++++++++++++++++++\n");
		System.err.println("Data: " + inPath);
		
		HistogramEval eval = new HistogramEval(getData(inPath),tempPath);
		MHistogram histogram  = null;
		
		System.err.println("Buckets " + numberOfBuckets);

		switch (histogram_type){
			case "i": 
				eval.buildMinSkewHist(numberOfBuckets*2, gridSize, true); // standard min skew 2^7 x 2^7 grid
				histogram = eval.getMinSkewHist();
				break;
			case "ii": 
				// standard min skew 2^7 x 2^7 grid and three refinerment steps
				eval.buildMinSkewProgressiveHist(numberOfBuckets*2, gridSize, 3, true);
				histogram = eval.getMinSkewProgressiveRefinementHistogram();
				break;
			}

		
		eval.dumpHistogram(histogram,getPrintStream(outPath));
		
		if (showPlot) 
			eval.showHist(histogram_type, histogram);
		
		
		System.err.println("Done.");
		System.err.println("++++++++++++++++++++++++++++++++++++\n");
	}

}