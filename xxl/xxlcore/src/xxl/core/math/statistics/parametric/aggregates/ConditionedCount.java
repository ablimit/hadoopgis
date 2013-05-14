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

import xxl.core.math.functions.AggregationFunction;
import xxl.core.predicates.Predicate;


public class ConditionedCount<I> extends AggregationFunction<I,Long> {

	protected Predicate<I> condition;
	
	public ConditionedCount(Predicate<I> condition) {
		this.condition = condition;
	}
	
	@Override
	public Long invoke(Long aggregate, I next) {
		return aggregate == null ?
			next == null ? 0 : 1 
			:
			next == null ? aggregate : ( condition.invoke(next) ? aggregate+1 : aggregate );
	}
}

