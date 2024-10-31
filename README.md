# MNIST Reader

Program designed to recognize digits from the MNIST dataset written in C++.

# How to run it:

Change these two values in main.cpp: TRAINING_PICS and TRAINING_LABELS to the paths of the MNIST pics and labels on your computer, compile and run.

# How it works:

The program randomly selects 10 digits from the dataset (one of each 0-9), trains on them until it reaches 100 percent accuracy or it has trained on them 100 times (whichever comes first). Then, it selects 10 digits at random again (the user is informed of this happening; the program will print New Values to the terminal) and repeats the process.

# Understanding the output:

> 0.179346 EPOCH OVER!   0.9 23 1

The first value is that of the cost function. The second (0.9) represents the models current accuracy, the third is the number of times it has trained on the current ten digits and the fourth is the number of times new digits have been selected.

# Proof that it is learning:

Initially, the models accuracy on a new set of digits will be quite low, but as it learns, its starting accuracy with digits its never seen will increase. 
