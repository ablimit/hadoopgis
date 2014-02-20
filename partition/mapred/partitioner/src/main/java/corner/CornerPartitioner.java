package abby;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;

import org.apache.hadoop.mapreduce.Partitioner;

/** Partition keys by their {@link Object#hashCode()}. */
public class CornerPartitioner<K, V> extends Partitioner<K, V> {
	public static void main(String[] args){}
    public static final ArrayList<Point> splitPoints = getSplitPoints("splitpoints");

    /** Use {@link Object#hashCode()} to partition. */
    public int getPartition(K key, V value, int numReduceTasks) {
      return (key.hashCode() & Integer.MAX_VALUE) % numReduceTasks;
    }

    public static ArrayList<Point> getSplitPoints(String p){
      ArrayList<Point> points = new ArrayList<Point>(100);
      try {
          BufferedReader br = new BufferedReader(new FileReader(p));
          String line = br.readLine();
          while (line != null) {
          String [] fields = line.split(" ");
          points.add(new Point(Double.parseDouble(fields[0]), Double.parseDouble(fields[1])));
          line = br.readLine();
        }
        br.close();
      }
      
      catch(Exception e)
      {
    	  System.err.println(e.getMessage());
      }
      return points;
    }

    public int findPartition(K key) {
      String k = key.toString();
      //final int pos = Arrays.binarySearch(splitPoints, key, comparator) + 1;
      //return (pos < 0) ? -pos : pos;
      return 0;
    }


    //ObjectInputStream is = new ObjectInputStream(new FileInputStream("targets"));
}
