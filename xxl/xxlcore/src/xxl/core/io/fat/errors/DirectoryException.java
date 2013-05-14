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

package xxl.core.io.fat.errors;

import java.io.IOException;

/**
 * This class is the top exception of this project. It indicates if there
 * was an exception which belongs to the DIR, FAT, BPB, FSI, and classes
 * which use them.
 */
public class DirectoryException extends IOException
{
	/**
	 * Create an instance of this object.
	 * @param str error message.
	 */
	public DirectoryException(String str)
	{
		super(str);
	}	//end constructor.

}	//end class DirectoryException
