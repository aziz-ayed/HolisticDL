# Distillation with Holistic Deep Learning

In this repository, we adapt the Holistic Deep Learning framework in order to make it compatible with distillation.
You can directly use it on the following datasets without further modifications: MNIST, CIFAR, and all of the UCI database.

To train a basic feed-forward neural network go to ```src``` and execute:

```python train.py --batch_range 64 --network_size 256 128 --stab_ratio_range 0.8 --l2 1e-5 --data_set uci10 --train_size 0.8 --lr 3e-4 --val_size 0.2 --n_distillations 1```


Sparsity, stability and robustness components from HDL are integrated, using the following parameters: ```--is_stable``` and tune ```--stab_ratio_range 0.8```.
To train a robust neural network, add: ```-r 1e-3```, and tune the parameter.
To train a sparse neural network, add: ```--l0 1e-5```, and tune the parameter.

Once you have trained your first, traditional neural network, you can iteratively proceed with distillation rounds adding the parameter ```--n_distillations 2```, and so on.  

Different metrics are automatically printed every 1000 training steps and at the end of each round: train and test accuracy, train val and test l2 loss against ground truth, gini stability, adversarial testing accuracy, percentage of zero weights, l2 loss against teacher model...

The ```outputs/distillation_DATASET``` contain the extracted soft-labels for each round of distillation.

The folder ```src``` contains:
- ```config.json```: the configuration file to determine some hyperparameters in your experiments.
- ```input_data.py```: the file to load and prepare the different datasets.
- ```pgd_attack```: implements the neural network attacks
- ```train.py```: runs the training. The most important file.
- ```Networks```: this folder contains the neural network architectures. We provide feed-forward and an old version of CNN_model.py that needs to be adapted to fit this code.

The folder ```utils``` contains:
- ```utils_print.py```: contains the printing functions.
- ```utils_model.py```: contains the loss functions and model dictionaries.
- ```utils_init.py```: contains function to load the different arguments.
- ```utils_MLP_model.py```: contains initialization function for MLP models. Needs to be adapted for CNNs.
- ```utils.py```: contains miscellaneous elements.

You can modify parameters in the config file with the desired values.

The documentation is a work in progress and will be updated soon.



