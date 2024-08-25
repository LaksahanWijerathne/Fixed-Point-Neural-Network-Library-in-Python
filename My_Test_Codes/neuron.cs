using System;

class Neuron
{
    private int[] weights;
    private int bias;
    private int scalingFactor;
    
    // Constructor to initialize the neuron
    public Neuron(int numInputs, int decimalPoints)
    {
        // Determine the scaling factor based on the desired number of decimal points
        scalingFactor = (int)Math.Pow(10, decimalPoints);

        // Initialize weights and bias with random integers
        weights = new int[numInputs];
        Random rand = new Random();
        for (int i = 0; i < numInputs; i++)
        {
            weights[i] = rand.Next(-scalingFactor, scalingFactor); // Random weights between -1 and 1 scaled
        }
        bias = rand.Next(-scalingFactor, scalingFactor); // Random bias between -1 and 1 scaled
    }

    // Method to calculate the neuron's output
    public int ComputeOutput(int[] inputs)
    {
        int weightedSum = 0;

        for (int i = 0; i < inputs.Length; i++)
        {
            weightedSum += weights[i] * inputs[i]; // Compute the weighted sum
        }

        weightedSum += bias;

        return ApplyActivationFunction(weightedSum);
    }

    // Activation function (ReLU in this case)
    private int ApplyActivationFunction(int sum)
    {
        // ReLU: Returns sum if it's positive, otherwise returns 0
        return sum > 0 ? sum : 0;
    }

    // Method to train the neuron using a single training example
    public void Train(int[] inputs, int target, double learningRate)
    {
        int output = ComputeOutput(inputs);
        int error = target - output;

        // Update weights and bias using the error and learning rate
        for (int i = 0; i < weights.Length; i++)
        {
            weights[i] += (int)(learningRate * error * inputs[i]);
        }

        bias += (int)(learningRate * error);
    }

    // Method to scale a floating-point input to the fixed-point representation
    public int ScaleInput(double input)
    {
        return (int)(input * scalingFactor);
    }

    // Method to convert fixed-point output back to floating-point
    public double ConvertOutput(int output)
    {
        return (double)output / scalingFactor;
    }
}

class Program
{
    static void Main(string[] args)
    {
        // Example usage: Create a neuron with 3 inputs and precision of 2 decimal points
        Neuron neuron = new Neuron(3, 2);

        // Example inputs
        double[] inputs = { 0.25, -0.75, 0.50 };
        int[] scaledInputs = Array.ConvertAll(inputs, x => neuron.ScaleInput(x));

        // Compute the neuron's output
        int output = neuron.ComputeOutput(scaledInputs);
        double actualOutput = neuron.ConvertOutput(output);

        Console.WriteLine($"Neuron Output: {actualOutput}");

        // Train the neuron with a target output (example target = 0.5)
        neuron.Train(scaledInputs, neuron.ScaleInput(0.5), 0.01);

        // Compute the output again after training
        output = neuron.ComputeOutput(scaledInputs);
        actualOutput = neuron.ConvertOutput(output);

        Console.WriteLine($"Neuron Output after Training: {actualOutput}");
    }
}
