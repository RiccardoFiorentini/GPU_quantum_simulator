#!/bin/bash

# Define the number of times to run the C++ program
NUM_RUNS=5

NUM_START=5
NUM_END=20

echo "Cuda Results" >>  OutputTest.txt

# Loop to run the C++ program multiple times
for((j = NUM_START; j <= NUM_END; j++)); do
    echo "Random num $j:" >> OutputTest.txt
    for ((i = 1; i <= NUM_RUNS; i++)); do
        #echo "Run $i:" >> results.txt   
        ./cudaExe "random_circs_ad/random_$j.qasm" >>  OutputTest.txt
    done
    echo "----------" >>  OutputTest.txt
done

echo "===================" >>  OutputTest.txt

echo "Cpp Results" >>  OutputTest.txt
for((j = NUM_START; j <= NUM_END; j++)); do
    echo "Random num $j:" >> OutputTest.txt
    for ((i = 1; i <= NUM_RUNS; i++)); do
        #echo "Run $i:" >> results.txt   
        ./CppExe "random_circs/random_$j.qasm" 1 >>  OutputTest.txt
    done
    echo "----------" >>  OutputTest.txt
done
