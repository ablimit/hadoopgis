#! /bin/bash

#if [ ! $# == 1 ]; then
    # echo "Usage: $0 [desktop | cluster | imac]"
    # exit
#fi

dir=testdir
testcase="gbm1.1.markup.ablet.1"

if [ -e ${dir}/q1.out.db ] && [ -e ${dir}/q2.out.db ] && [ -e ${dir}/q3.out.db ] && [ -e ${dir}/q4.out.db ] && [ -e ${dir}/gbm1.1.markup.ablet.1 ]
then
    echo "All testing infrastructure is present."
else 
    echo "Test cases are missing... Can't do test without them."
    exit 1
fi

#case "$1" in
#    desktop) echo "Test begin in $1"
#    mymakefile="Makefile"
#    ;;
#    cluster) echo "Test begin in $1"
#    mymakefile="makefile"
#    ;;
#    imac) echo "Test begin in $1"
#    mymakefile="MacMakefile"
#    ;;
#    *) echo "[$1] -- Unknown environment. "
#    exit 1
#esac

# create the test files.
make -f Makefile
export map_input_file="gbm1.1"

# test 1 
echo -n "TEST: Query 1 --- "

./q1 < ${dir}/${testcase} | sort > ${dir}/q1.out.mr

diff ${dir}/q1.out.db ${dir}/q1.out.mr >/dev/null 2>&1

if [ $? -ne 0 ]
then
    echo "failed."
else
    echo "passed."
    rm ${dir}/q1.out.mr
fi

# test 2 
echo -n "TEST: Query 2 --- "

./q2 < ${dir}/${testcase} | sort > ${dir}/q2.out.mr

diff ${dir}/q2.out.db ${dir}/q2.out.mr >/dev/null 2>&1

if [ $? -ne 0 ]
then
    echo "failed."
else
    echo "passed."
    rm ${dir}/q2.out.mr
fi

# test 3
echo -n "TEST: Query 3 --- "

./q3 < ${dir}/q3.dat | sort > ${dir}/q3.out.mr

diff ${dir}/q3.out.db ${dir}/q3.out.mr >/dev/null 2>&1

if [ $? -ne 0 ]
then
    echo "failed."
else
    echo "passed."
    rm ${dir}/q3.out.mr
fi

# test 4
echo -n "TEST: Query 4 --- "

./q4 < ${dir}/gbm1.1.markup.ablet.1 | sort > ${dir}/q4.out.mr

diff ${dir}/q4.out.db ${dir}/q4.out.mr >/dev/null 2>&1

if [ $? -ne 0 ]
then
    echo "failed."
else
    echo "passed."
    rm ${dir}/q4.out.mr
fi

# clean up the trash.
make clean

