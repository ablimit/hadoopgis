package corner;

public class Point implements Comparable<Point> {
	private double x;
	private double y;

	public Point(double x, double y) {
		super();
		this.x = x;
		this.y = y;
	}

	public int compareTo(Point p) {
		if (this.x < p.x) {
			//System.err.println("x:"+this.x + "<" + p.x);
			return -1;
		} else if (this.x > p.x) {
			//System.err.println("x:"+this.x + ">" + p.x);
			return 1;
		} else {
			if (this.y < p.y) {
				//System.err.println("y:"+this.y + "<" + p.y);
				return -1;
			} else if (this.y > p.y) {
				//System.err.println("y:"+this.y + ">" + p.y);
				return 1;
			} else {
				//System.err.println(this.x + "," + this.y + "==" + p.x + ","
						//+ p.y);
			}
			return 0;
		}
	}
}
