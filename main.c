#include <stdio.h>
#include <stdlib.h>
#include "include/Core/training.h"
#include "include/Core/dataset.h"
#include "include/Core/logging.h"

int main()
{
    set_log_level(LOG_LEVEL_INFO);
    
    NeuralNetwork *network = create_neural_network(2);
    build_network(network, OPTIMIZER_ADAM, 0.1f, LOSS_MSE, 0.05f, 0.01f);
    model_add(network, LAYER_DENSE, ACTIVATION_RELU, 5, 52, 0.0f, 0, 0);
    model_add(network, LAYER_DENSE, ACTIVATION_TANH, 52, 52, 0.0f, 0, 0);
    model_add(network, LAYER_DENSE, ACTIVATION_SIGMOID, 52, 52, 0.0f, 0, 0);
model_add(network, LAYER_DENSE, ACTIVATION_TANH, 52, 52, 0.0f, 0, 0);
model_add(network, LAYER_DENSE, ACTIVATION_TANH, 52, 1, 0.0f, 0, 0);
    float X_data[52][5];
    for(int i =0;i<52;i++)
    {
        for(int j = 0;j<5;j++)
        {
            X_data[i][j]=(1.0f*(1+rand()%(10000))*(1+rand()%(10000))) / 100000000.0f;
        }
    }

    float y_data[52][1];
    for(int i =0;i<5;i++)
    {
        float val =0;
        for(int j =0;j<52;j++)
        {
            val +=(1.f*(j+1))*X_data[i][j];
        }
        y_data[i][0] = val/(10000.f);
    }

    Dataset *dataset = dataset_create();
    dataset_load_arrays(dataset, (float *)X_data, (float *)y_data, 52, 100, 1);

    summary(network);

    train_network(network, dataset, 1000);
    test_network(network, dataset->X, dataset->y, dataset->num_samples, NULL);

    dataset_free(dataset);
    free_neural_network(network);

    return 0;
}