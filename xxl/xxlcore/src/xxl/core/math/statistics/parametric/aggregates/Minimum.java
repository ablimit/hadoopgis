/* XXL: The eXtensible and fleXible Library for data processing

Copyright (C) 2000-2011 Prof. Dr. Bernhard Seeger
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

package xxl.core.math.statistics.parametric.aggregates;

import java.util.Comparator;

import xxl.core.comparators.ComparableComparator;
import xxl.core.math.functions.AggregationFunction;

/**
 * Computes the minimum of given data without any error control wrt.&nbsp;to a given ordering imposed
 * by a {@link java.util.Comparator comparator}.
 * <br>
 * <p><b>Objects of this type are recommended for the usage with aggregator cursors!</b></p>
 * <br>
 * Each aggregation function must support a function call of the following type:<br>
 * <tt>agg_n = f (agg_n-1, next)</tt>, <br>
 * where <tt>agg_n</tt> denotes the computed aggregation value after <tt>n</tt> steps,
 * <tt>f</tt> the aggregation function,
 * <tt>agg_n-1</tt> the computed aggregation value after <tt>n-1</tt> steps
 * and <tt>next</tt> the next object to use for computation.
 * An aggregation function delivers only <tt>null</tt> as aggregation result as long as the aggregation
 * function has not yet fully initialized.
 * <br>
 * Objects of this class don't use any internally stored information to obtain the minimum, 
 * so one could say objects of this type are 'status-less'.
 * See {@link xxl.core.math.statistics.parametric.aggregates.OnlineAggregation OnlineAggregation} for further details about 
 * aggregation function using internally stored information.
 *
 * Consider the following example:
 * <code><pre>
 * Aggregator agg = new Aggregator(
		new DiscreteRandomNumber(new JavaDiscreteRandomWrapper(100), 50), // input-Cursor
		new Minimum()	// aggregate function
	);
 * <\code><\pre>
 * <br>
 *
 * @see xxl.core.cursors.mappers.Aggregator
 * @see xxl.core.functions.Function
 * @see java.util.Comparator
 * @see xxl.core.math.statistics.parametric.aggregates.OnlineAggregation
 */

public class Minimum extends AggregationFunction<Number,Number> {

	/** comparator imposing the total ordering used for determining the minimum of the given data */
	protected Comparator comparator;

	/** Constructs a new object of this class.
	 * 
	 * @param comparator that imposes the total ordering used for determining the minimum of the given data
	 */
	public Minimum(Comparator comparator) {
		this.comparator = comparator;
	}

	/** Constructs a new object of this class using the 'natural ordering' of the treated objects.
	 */
	public Minimum() {
		this(new ComparableComparator());
	}

	/** Two-figured function call for supporting aggregation by this function.
	 * Each aggregation function must support a function call like <tt>agg_n = f (agg_n-1, next)</tt>,
	 * where <tt>agg_n</tt> denotes the computed aggregation value after <tt>n</tt> steps, <tt>f</tt>
	 * the aggregation function, <tt>agg_n-1</tt> the computed aggregation value after <tt>n-1</tt> steps
	 * and <tt>next</tt> the next object to use for computation.
	 * This method delivers only <tt>null</tt> as aggregation result as long as the aggregation
	 * has not yet initialized.
	 * 
	 * @param min result of the aggregation function in the previous computation step (minimum object so far)
	 * @param next next object used for computation
	 * @return aggregation value after n steps
	 */
	public Number invoke(Number min, Number next) {
		return next == null ? min : min == null ? next.doubleValue() : Math.min(min.doubleValue(), next.doubleValue());
	}
}
