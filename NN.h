double sigmoid(double inp)
{
    return(1.0/(1.0 + exp(-1.0 * inp)));
}

double sigmoidD(double inp)
{
    return(sigmoid(inp) * (1.0 - sigmoid(inp)));
}

class Neuron
{
    public:
    double valZ = 0.0;//pre sigmoid
    double valA = 0.0;//post sigmoid
    double bias;
    double* weights;
    double* weightsChangesPP;//pre procvessing

    double biasChangePP;

    void init(int numInNext)
    {
        weights = (double*)malloc(numInNext * sizeof(double));
        weightsChangesPP = (double*)malloc(numInNext * sizeof(double));

        bias = ((double)rand()/(double)RAND_MAX) * 0.1;
        bias -= 0.05;
        biasChangePP = 0.0;
        //bias = 0;

        for(int i = 0; i < numInNext; i++)
        {            
            weights[i] = ((double)rand()/(double)RAND_MAX) * 0.1;
            weights[i] -= 0.05;
            weightsChangesPP[i] = 0.0;
            //weights[i] = 0;
        }
    }
};

class Layer
{    
    public:
    Neuron* neurons;
    int numNeurons;

    void init(int numNeuronsIL, int numInNext)
    {
        numNeurons = numNeuronsIL;
        neurons = (Neuron*)malloc(sizeof(Neuron) * numNeuronsIL);

        for(int i = 0; i < numNeurons; i++)
        {
            neurons[i].init(numInNext);
        }
    }
};

class Network
{
public:
    void init(int* sizesOfLayers, uint8_t nL)
    {
        numLayers = nL;
        layer = (Layer*)malloc(sizeof(Layer) * numLayers);

        for(int i = 0; i < numLayers; i++)
        {
            layer[i].init(sizesOfLayers[i], (i == numLayers - 1)?0:sizesOfLayers[i + 1]);
        }
    }

    void setFirstLayer(double* inp)
    {
        for(int i = 0; i < layer[0].numNeurons; i++)
        {
            layer[0].neurons[i].valA = inp[i];
        }
    }

    void backProp(Layer dL)
    {
        // 0 - 1 - 2
        for(int i = 0; i < layer[1].numNeurons; i++)//do  1 - 2 layer inter weights + biases first. theyre easy
        {
            for(int j = 0; j < layer[2].numNeurons; j++)
            {
                layer[2].neurons[j].biasChangePP += 2 * (layer[2].neurons[j].valA - dL.neurons[j].valA) * sigmoidD(layer[2].neurons[j].valZ); 
                layer[1].neurons[i].weightsChangesPP[j] += 2 * (layer[2].neurons[j].valA - dL.neurons[j].valA) * sigmoidD(layer[2].neurons[j].valZ) 
                * layer[1].neurons[i].valA;
            }
        }

        for(int a = 0; a < layer[0].numNeurons; a++)
        {
            for(int b = 0; b < layer[1].numNeurons; b++)
            {
                for(int c  = 0; c < layer[2].numNeurons; c++)
                {
                    //[0].valA is incorrct. we would be adding it up for each output neuron
                    layer[0].neurons[a].weightsChangesPP[b] += 2 * (layer[2].neurons[c].valA - dL.neurons[c].valA) * sigmoidD(layer[2].neurons[c].valZ) 
                    * layer[1].neurons[b].weights[c] * sigmoidD(layer[1].neurons[b].valZ) * layer[0].neurons[a].valA;

                    //[0].valA is incorrct. we would be adding it up for each output neuron
                    if(!a)
                    {
                        layer[1].neurons[b].biasChangePP += 2 * (layer[2].neurons[c].valA - dL.neurons[c].valA) * sigmoidD(layer[2].neurons[c].valZ) 
                        * layer[1].neurons[b].weights[c] * sigmoidD(layer[1].neurons[b].valZ);
                    }                    
                }              
            }
        }
    }

    void commit(double learningRate, int numExamples)
    {
        for(int i = 0; i < numLayers; i++)
        {
            for(int j = 0; j < layer[i].numNeurons; j++)
            {
                if(i + 1 != numLayers)
                {
                    for(int k = 0; k < (layer[i + 1].numNeurons); k++)
                    {     
                        //std::cout << i << " " << j << " " << k << std::endl;;               
                        layer[i].neurons[j].weights[k] -= (( layer[i].neurons[j].weightsChangesPP[k]/(double)numExamples) * learningRate);
                        layer[i].neurons[j].weightsChangesPP[k] = 0;
                    }
                }
               
                layer[i].neurons[j].bias -= ((layer[i].neurons[j].biasChangePP/(double)numExamples) * learningRate);
                layer[i].neurons[j].biasChangePP = 0;
            }
        }
    }

    Layer compute()
    {
        for(int i = 1; i < numLayers; i++)
        {
            for(int j = 0; j < layer[i].numNeurons; j++)
            {
                layer[i].neurons[j].valZ = layer[i].neurons[j].bias;
                layer[i].neurons[j].valA = 0.0;
                for(int k = 0; k < layer[i - 1].numNeurons; k++)
                {
                    layer[i].neurons[j].valZ += (layer[i - 1].neurons[k].valA * layer[i - 1].neurons[k].weights[j]);
                }

                layer[i].neurons[j].valA = sigmoid(layer[i].neurons[j].valZ);
            }
        }

        return(layer[numLayers - 1]);
    }

    private:
    uint8_t numLayers;
    Layer* layer;
};