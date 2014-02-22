import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;

import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.Partitioner;
import org.apache.hadoop.io.Text;

public class CornerPartitioner<K, V> implements Partitioner<K, V> {

	private ArrayList<Point> splitPoints;

	public CornerPartitioner() throws IOException{
		super();
		// TODO Auto-generated constructor stub
		splitPoints = getSplitPoints("splitpoints.dat");
	}
	
	@Override
	public int getPartition(K key, V value, int numReduceTasks) {
		return (this.findPartition(key) % numReduceTasks);
	}

	public static ArrayList<Point> getSplitPoints(String p) throws IOException {
		ArrayList<Point> points = new ArrayList<Point>(100);
		BufferedReader br = new BufferedReader(new FileReader(p));
		String line = br.readLine();
		while (line != null) {
			String[] fields = line.split(" ");
			/*
			System.err.println(Double.parseDouble(fields[0])+ "," + Double
					.parseDouble(fields[1])); */
			points.add(new Point(Double.parseDouble(fields[0]), Double
					.parseDouble(fields[1])));
			line = br.readLine();
		}
		br.close();
		return points;
	}

	public int findPartition(K key) {
		String k = key.toString();
		String[] fields = k.split(" ", 2);
		double[] coor = {Double.parseDouble(fields[0]),
				Double.parseDouble(fields[1])};
		final int pos = Collections.binarySearch(this.splitPoints, new Point(
				coor[0], coor[1]))+1;
		return (pos < 0) ? -pos : pos;
	}

	public static void main(String[] args) throws IOException {
		/*
		Point a = new Point(0.1, 0.2);
		Point b = new Point(0.1, 0.3);
		Point c = new Point(0.2, 0.2);
		System.err.println(a.compareTo(a));
		System.err.println(a.compareTo(b));
		System.err.println(a.compareTo(c));
		System.err.println(b.compareTo(a));
		System.err.println(b.compareTo(b));
		System.err.println(b.compareTo(c));
		System.err.println(c.compareTo(a));
		System.err.println(c.compareTo(b));
		System.err.println(c.compareTo(c));
		System.exit(0);*/
		Text key = new Text("0.5 0.5");
		if (args.length ==2)
			key.set(args[0].trim() + " " + args[1].trim());
		CornerPartitioner<Text, Text> part = new CornerPartitioner<Text, Text>();
		System.out.println(part.findPartition(key));
	}

	@Override
	public void configure(JobConf arg0) {
	}
}
