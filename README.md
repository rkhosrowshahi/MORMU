# Multi-Objective Responsible Machine Unlearning

In this project, we aim to propose a novel machine unlearning method for black-box neural networks such as Multi-layer Perceptron (MLP) and ConvNets (LeNet-5) using evolutionary multi-objective algorithm. We used non-dominated sorting genetic algorithm II (NSGA-II) []which is a meta-heuristic gradient-free algorithm to effectively optimize $`M \ge 2`$ objectives with respect to parameters such as weights and biases in neural networks.

The datasets used:

* Fashion-MNIST

There are two dataset splits in Fashion known as:

1. Training data $`D_{train}`$
2. Test data $`D_{test}`$

For the sake of unlearning, the training data is divided into two subsets as follows:

1. Forget data $`D_{u}`$
2. Retaining data $`D_{r}`$

The forgetting data is randomly selected from training data and the size is 1,000.

There are two networks to use and suggested commands are as follows:

1. MLP (64 hidden units), 50K params

    ```
    python3 main.py --net mlp --dataset fashion --output_dir ./out/mlp-64 --steps 1000
    ```

2. LeNet-5, 60K params

    ```
    python3 main.py --net lenet --dataset fashion --output_dir ./out/lenet --steps 1000
    ```
