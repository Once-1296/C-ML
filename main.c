#include <stdio.h>
#include <stdlib.h>
#include "include/Core/training.h"
#include "include/Core/dataset.h"
#include "include/Core/logging.h"

int main()
{
    set_log_level(LOG_LEVEL_INFO);
    int A=4,B=2,C=1;
    NeuralNetwork *network = create_neural_network(2);
    build_network(network, OPTIMIZER_ADAM, 0.1f, LOSS_MSE, 0.05f, 0.01f);
    model_add(network, LAYER_DENSE, ACTIVATION_TANH, B, A, 0.0f, 0, 0);
    model_add(network, LAYER_DENSE, ACTIVATION_TANH, A, A, 0.0f, 0, 0);
model_add(network, LAYER_DENSE, ACTIVATION_TANH, A, C, 0.0f, 0, 0);

    float X_data[A][B];
    for(int i =0;i<A;i++)
    {
        for(int j = 0;j<B;j++)
        {
            X_data[i][j]=(1.0f*(1+rand()%(10000))*(1+rand()%(10000))) / 100000000.0f;
        }
    }

    float y_data[A][C];
    for(int i =0;i<B;i++)
    {
        float val =0;
        for(int j =0;j<A;j++)
        {
            val +=(1.f*(j+1))*X_data[i][j];
        }
        y_data[i][0] = val/(100.f);
    }

    Dataset *dataset = dataset_create();
    dataset_load_arrays(dataset, (float *)X_data, (float *)y_data, A, B, 1);

    summary(network);

    train_network(network, dataset, 1000);
    test_network(network, dataset->X, dataset->y, dataset->num_samples, NULL);

    dataset_free(dataset);
    free_neural_network(network);

    return 0;
}