package abby;

public class Point implements Comparable<Point>{
    private double x;
    private double y;
    public Point(double x, double y) {
      super();
      this.x = x;
      this.y = y;
    }

    public int compareTo(Point p) {
      if (this.x < p.x ) 
        return 1; 
      else if (this.x > p.x)
        return -1; 
      else 
      {
      if (this.y < p.y ) 
        return 1; 
      else if (this.y > p.y)
        return -1; 
      else 
        return 0;
      }
    }
  }

