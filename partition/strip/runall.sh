#! /bin/bash

sh x.osm.pipeline.sh
rm c*.txt

sh x.pais.pipeline.sh
rm c*.txt

sh y.osm.pipeline.sh
rm c*.txt

sh y.pais.pipeline.sh
rm c*.txt
