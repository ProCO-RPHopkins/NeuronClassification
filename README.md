# Neuron Classification Using Neural Networks in Julia

## Table of Contents

1. Project Overview
    * Neuron Firing Patterns
        * Firing Rate
        * Neuron Types
    * Example Data
2. What Happens When You Run the Code
    * Generate Data
    * Load and Explore Data
    * Prepare Data for Training
    * Build the Neural Network
    * Train the Neural Network
    * Test the Neural Network
3. Output
4. Usage
5. Contributing
6. In Summary
7. License

## Project Overview

This project demonstrates how to classify neurons based on their firing patterns using a simple neural network implemented in Julia. It covers various aspects of data science, machine learning, and computational neuroscience.

### Neuron Firing Patterns

1. Firing Rate
    * The firing rate is a number that represents how frequently a neuron fires or sends signals. In the synthetic dataset, this value is randomly chosen between 50 and 150.
    * For example, a firing rate of 75 means the neuron fires 75 times within a certain period.
2. Neuron Types
    * Each neuron is classified as either “Excitatory” or “Inhibitory”.
    * Excitatory Neurons - These neurons increase the likelihood of the neurons they connect to firing. They typically have higher firing rates.
    * Inhibitory Neurons - These neurons decrease the likelihood of the neurons they connect to firing. They typically have lower firing rates.

### Example Data

Here's an example of the generated neuron data

| ID | Firing Rate |    Type    |
|----|-------------|------------|
| 1  | 120         | Excitatory |
| 2  | 85          | Inhibitory |
| 3  | 140         | Excitatory |
| 4  | 60          | Inhibitory |
| 5  | 130         | Excitatory |

#### In this table
    * id: A unique identifier for each neuron.
    * firing_rate: The number of times the neuron fires within a given period.
    * class: The type of neuron (either “Excitatory” or “Inhibitory”)

## What Happens When You Run the Code

1. **Generate Data**
   * The code first creates a fake dataset of neurons. Each neuron has a unique ID, a firing rate (how often it sends signals), and a type (either "Excitatory" or "Inhibitory").
   * This data is saved in a file called `neuron_data.csv`.

2. **Load and Explore Data**
   * The code then reads this data from the file and shows you the first few rows to give you an idea of what it looks like.
   * It also provides some basic statistics about the data, like the average firing rate and the number of each type of neuron.

3. **Prepare Data for Training**
   * The neuron types are converted into a format that the neural network can understand (numbers instead of words).
   * The data is split into two parts: one for training the neural network and one for testing it later.

4. **Build the Neural Network**
   * A simple neural network is created. Think of this as a virtual brain that will learn to recognize the patterns in the neuron data.

5. **Train the Neural Network**
   * The training data is fed into the neural network. The network adjusts its internal settings to learn how to classify neurons based on their firing rates.
   * This process involves calculating how wrong the network's guesses are and making corrections to improve accuracy.

6. **Test the Neural Network**
   * The testing data is used to see how well the neural network learned to classify neurons.
   * The network makes predictions on the test data, and these predictions are compared to the actual neuron types to calculate accuracy.

## Output

When you run the code, you will see

* **Initial Data**: A preview of the first few rows of the generated neuron data.

* **Data Statistics**: Summary statistics of the dataset.

* **Training Progress**: Information about the training process (this might include loss values, which indicate how well the network is learning).

* **Model Accuracy**: The final accuracy of the neural network on the test data, which tells you how well the network can classify neurons based on their firing rates.

For example, you might see an output like:

```sh
Model Accuracy: 0.85
```

This means the neural network correctly classified 85% of the neurons in the test data.

## Usage

1. **Run the Jupyter Notebook**: Start Jupyter Notebook and open the project notebook:

   ```sh
   jupyter notebook
   ```

2. **Follow the Steps**: Execute the cells in the notebook to run the project step-by-step.

## Contributing

Contributions are welcome! Please fork this repository and submit a pull request with your improvements.

## In Summary

This project helps you understand how to use Julia to create a neural network that can classify neurons. It involves generating data, training a model, and evaluating its performance. The final output is a measure of how accurately the model can classify new neurons based on their firing patterns.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
