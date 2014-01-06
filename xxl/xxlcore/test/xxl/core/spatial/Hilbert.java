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

import xxl.core.spatial.SpaceFillingCurves;


public class Hilbert{
  static public int FILLING_CURVE_PRECISION ;

  public static long getHilbertValue(double x, double y) {
    return SpaceFillingCurves.hilbert2d((int) (x*FILLING_CURVE_PRECISION),(int) (y*FILLING_CURVE_PRECISION));
  }

  public static void dojob(String path) throws IOException {
    BufferedReader br = new BufferedReader(new FileReader (path));
    String line ;
    String space = " " ;
    double[] leftCorner = new double[2];
    double[] rightCorner = new double[2];
    double[] center = new double[2];
    long [] hilbert = new long [3];

    while (null != (line = br.readLine()))
    {
      String[] sp = line.split("\\s+");
      if (sp.length >= 5) {
        leftCorner[0] = Double.parseDouble(sp[1]);
        leftCorner[1] = Double.parseDouble(sp[2]);

        rightCorner[0] = Double.parseDouble(sp[3]);
        rightCorner[1] = Double.parseDouble(sp[4]);

        center[0] = (leftCorner[0] + rightCorner[0])/2;
        center[1] = (leftCorner[1] + rightCorner[1])/2;

        hilbert [0] = getHilbertValue(leftCorner[0],leftCorner[1]);
        hilbert [1] = getHilbertValue(center[0],center[1]);
        hilbert [2] = getHilbertValue(rightCorner[0],rightCorner[1]);
        System.out.println(line + space + hilbert[0] + space + hilbert[1] + space + hilbert[2]);
      }
    }

    br.close();
  }


  public static void main(String[] args) throws IOException {
    if (args.length <2 )
    {
      System.err.println("Usage: "
          + Hilbert.class.getSimpleName() 
          +"[input Data] [HilbertCurvePrecision]");
      System.exit(0);
    }

    String inPath = args[0]; // data path
    FILLING_CURVE_PRECISION = 1<<Integer.parseInt(args[1]); // precision 

    dojob(inPath);

    System.err.println("Done.");
    System.err.println("++++++++++++++++++++++++++++++++++++\n");
  }

}
