#! /bin/bash

sh osm.pipeline.sh
rm c*.txt

sh pais.pipeline.sh
rm c*.txt
