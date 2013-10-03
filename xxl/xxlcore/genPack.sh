#! /bin/bash

mvn clean

#compile without running tests
mvn -DskipTests=true package

# generate test jar
mvn jar:test-jar

