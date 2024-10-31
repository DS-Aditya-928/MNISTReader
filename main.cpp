#include <iostream>
#include <fstream>
#include <intrin.h>
#include <cmath>
#include "NN.h"
#define TRAINING_PICS "C:\\Users\\Aditya.D.S\\Downloads\\train-images-idx3-ubyte\\train-images.idx3-ubyte"
#define TRAINING_LABELS "C:\\Users\\Aditya.D.S\\Downloads\\train-labels-idx1-ubyte\\train-labels.idx1-ubyte"

#define numSamplesPerEpoch 20

long double total = 0;
long double right = 0;

uint8_t trainData[10][28 * 28];
// uint8_t trainLabels[10];

// keep neuron vals between 0 and 1
int main()
{
    Network NN;
    srand(time(NULL));
    int sizeArr[3] = {28 * 28, 15, 10};

    NN.init(sizeArr, 3);

    char *conPixels[5] = {" ", "\u2591", "\u2592", "\u2593", "\u2588"};
    std::cout << "Comp Vis Test" << std::endl;
    std::ifstream myfile(TRAINING_LABELS, std::ios::binary);
    std::ifstream tpics(TRAINING_PICS, std::ios::binary);

    if (myfile.is_open() && tpics.is_open())
    {
        std::cout << "FILES OPENED!" << std::endl;
        int32_t magicNum = 0;
        myfile.read((char *)&magicNum, 4);
        std::cout << _byteswap_ulong((int32_t)magicNum) << std::endl;

        char *buf = (char *)malloc(16);
        tpics.read(buf, 16);
        free(buf);

        int32_t numEntries = 0;
        myfile.read((char *)&numEntries, 4);
        numEntries = _byteswap_ulong((int32_t)numEntries);
        std::cout << "NUM ENTRIES " << numEntries << std::endl;

        for (int x = 0; x < 10000; x++)
        {
            // grab 10 numbers. start from a random search index. find 10 vals.
            int randomIndex = (rand() / (float)RAND_MAX) * 59000.f;

            for (int a = 0; a < 10; a++)
            {
                myfile.seekg(randomIndex + 8, std::ios_base::beg);

                int bC = 0;

                while (true)
                {
                    int8_t label;
                    myfile.read((char *)&label, 1);

                    if (label == a)
                    {
                        tpics.seekg(((bC + randomIndex) * 28 * 28) + 16, std::ios_base::beg);
                        // uint8_t bufDat[28 * 28];
                        tpics.read((char *)trainData[a], 28 * 28);
                        break;
                    }
                    bC++;
                }

                /*
                for (int j = 0; j < 28; j++)
                {
                    for (int k = 0; k < 28; k++)
                    {
                        double val;
                        // val = tpics.get();
                        val = trainData[a][(j * 28) + k];
                        std::cout << conPixels[(int)((val / 255.f) * 4)];
                    }

                    std::cout << std::endl;
                }

                std::cout << a << std::endl;
                */
            }

            for (int repCounter = 0; repCounter < 100; repCounter++)
            { // train on the same data
                double overallCost = 0;
                right = 0;
                total = 0;

                for (int i = 0; i < 10; i++)
                {
                    double inpArr[28 * 28];
                    for (int l = 0; l < 28 * 28; l++)
                    {
                        inpArr[l] = ((double)trainData[i][l]) / 255.0;
                    }

                    NN.setFirstLayer(inpArr);

                    /*
                    for(int j = 0; j < 28; j++)
                    {
                        for(int k = 0; k < 28; k++)
                        {
                            double val;
                            //val = tpics.get();
                            val = inpArr[(j * 28) + k];
                            std::cout << conPixels[(int)(val*4)];
                        }

                        std::cout << std::endl;
                    }
                    */

                    // std::cout << "\n\n\n" << std::endl;
                    Layer outputNeurons; // = (Neuron*)malloc(10* sizeof(Neuron));
                    Layer desiredOutput;
                    desiredOutput.init(10, 0);
                    desiredOutput.neurons[i].valA = 1.0;
                    outputNeurons = NN.compute();

                    double miniCost = 0;

                    int biggest = 0;
                    for (int l = 0; l < 10; l++)
                    {
                        if (outputNeurons.neurons[l].valA > outputNeurons.neurons[biggest].valA)
                        {
                            biggest = l;
                        }

                        if (i == numSamplesPerEpoch - 1)
                        {
                            // std::cout << outputNeurons.neurons[l].valA << std::endl;
                        }
                        miniCost += ((outputNeurons.neurons[l].valA - desiredOutput.neurons[l].valA) * (outputNeurons.neurons[l].valA - desiredOutput.neurons[l].valA));
                    }

                    if (i == biggest)
                    {
                        right += 1;
                    }

                    total += 1;

                    overallCost += miniCost;
                    NN.backProp(desiredOutput);
                    // std::cout << miniCost << "\n\n\n\n\n\n" << std::endl;
                }

                overallCost = overallCost / numSamplesPerEpoch;
                std::cout << overallCost << " EPOCH OVER! "
                          << "  " << right / total << " " << repCounter << " " << x << std::endl;

                if(right / total == 1)
                {
                    break;
                }
                if(x < 100)
                {
                    NN.commit(1, 10);
                }
                
            }

            // cycle over. adjust vals and go again
            std::cout << "New Values" << std::endl;
        }
    }

    // while(true);
    return (0);
}