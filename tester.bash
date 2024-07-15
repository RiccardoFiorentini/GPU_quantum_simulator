#!/bin/bash

# Define the number of times to run
NUM_RUNS=5

#Loop to run the C++ program multiple times
echo "C version:" >> OverallTest.csv
for((j = 5; j <= 18; j++)); do
    echo "Num QBit $j:" >>  OverallTest.csv
    for ((i = 1; i <= NUM_RUNS; i++)); do  
        ./CExe "random_circs_ad/random_$j.qasm" >> OverallTest.csv
    done
done

#Scegli numero di thread ottimale
echo "Cuda Without Preprocessing:" >> OverallTest.csv
for((j = 5; j <= 22; j++)); do
    echo "Num QBit $j:" >> OverallTest.csv
    for ((i = 1; i <= NUM_RUNS; i++)); do
        ./cudaExe1 "random_circs_ad/random_$j.qasm" >>  OverallTest.csv
    done
done

#Scegli numero di thread ottimale
echo "Cuda With Preprocessing:" >> OverallTest.csv
for((j = 5; j <= 22; j++)); do
    echo "Num QBit $j:" >> OverallTest.csv
    for ((i = 1; i <= NUM_RUNS; i++)); do
        ./cudaExe2 "random_circs_ad/random_$j.qasm" >>  OverallTest.csv
    done
done

echo "Cuda With Preprocessing merged matrix:" >> OverallTest.csv
for((j = 5; j <= 22; j++)); do
    echo "Num QBit $j:" >> OverallTest.csv
    for ((i = 1; i <= NUM_RUNS; i++)); do
        ./cudaExe3 "random_circs_ad/random_$j.qasm" >>  OverallTest.csv
    done
done

echo "Cuda With Preprocessing read only:" >> OverallTest.csv
for((j = 5; j <= 22; j++)); do
    echo "Num QBit $j:" >> OverallTest.csv
    for ((i = 1; i <= NUM_RUNS; i++)); do
        ./cudaExe4 "random_circs_ad/random_$j.qasm" >>  OverallTest.csv
    done
done