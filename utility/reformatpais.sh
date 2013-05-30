#! /bin/bash

perl -p -i -e 's/,\+0/|/g' *.2 
perl -p -i -e 's/\.,"/|POLYGON((/g' *.2
perl -p -i -e 's/"/))/g' *.2
