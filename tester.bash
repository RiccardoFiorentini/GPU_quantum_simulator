#!/bin/bash

# Define the number of times to run the C++ program
NUM_RUNS=5

NUM_START=5
NUM_END=22

#Loop to run the C++ program multiple times
for((j = 19; j <= 22; j++)); do
    echo "Random num $j:" >>  OutputTest1.txt
    for ((i = 1; i <= NUM_RUNS; i++)); do  
        ./cudaExe2 "random_circs_ad/random_$j.qasm" >>   OutputTest1.txt
    done
    echo "----------" >>  OutputTest1.txt
done

#echo "Cuda Without Preprocessing:" >>   OutputCUDAOld.txt
#for((j = NUM_START; j <= NUM_END; j++)); do
#    echo "Random num $j:" >> OutputCUDAOld.txt
#    for ((i = 1; i <= NUM_RUNS; i++)); do
#        #echo "Run $i:" >> results.txt   
#        ./cudaExe1 "random_circs_ad/random_$j.qasm" >>  OutputCUDAOld.txt
#    done
#    echo "----------" >>  OutputCUDAOld.txt
#done
#
#echo "Cpp Results" >>  OutputCPP.txt
#for((j = NUM_START; j <= 14; j++)); do
#    echo "Random num $j:" >> OutputCPP.txt
#    for ((i = 1; i <= NUM_RUNS; i++)); do
#        #echo "Run $i:" >> results.txt   
#        ./CppExe "random_circs/random_$j.qasm" 1 >>  OutputCPP.txt
#    done
#    echo "----------" >>  OutputCPP.txt
#done
