#!/bin/bash

#./run_main.sh True 50 7 1.0
#./run_main.sh True 100 7 1.0
for i in 0 1 2
do
    ./run_main.sh True 200 3 1.0 $i
    ./run_main.sh True 200 7 1.0 $i
    ./run_main.sh True 200 11 1.0 $i
    ./run_main.sh True 200 15 1.0 $i
    ./run_main.sh True 200 19 0.5 $i
    ./run_main.sh True 200 19 1.0 $i
    ./run_main.sh True 200 19 1.5 $i
done