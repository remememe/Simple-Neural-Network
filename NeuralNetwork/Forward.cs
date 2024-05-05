using Microsoft.Win32;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Reflection.Emit;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using static System.Net.Mime.MediaTypeNames;
namespace NeuralNetwork
{
    internal class Forward
    {
        public Neuron[][] Neurons;
        public double[][,] Weights;
        private Random Rnd = new Random();
        private bool L1 = true, L2 = false;
        private double Lambda = 0.006;
        public double good = 1, bad = 1;
        public double acc = 0;
        public double[][] FlatImage(ref Bitmap Image)
        {
            int width = Image.Width;
            int height = Image.Height;
            Neurons = new Neuron[3][];
            Neurons[0] = new Neuron[width * height];
            double[][] FlatedImg = new double[width][];

            for (int i = 0; i < width; i++)
            {
                FlatedImg[i] = new double[height];
                for (int j = 0; j < height; j++)
                {
                    int index = i * height + j;
                    Neurons[0][index] = new Neuron();
                    Color pixelColor = Image.GetPixel(i, j);
                    int grayValue = (int)(0.299 * pixelColor.R + 0.587 * pixelColor.G + 0.114 * pixelColor.B);
                    Neurons[0][index].input = NormalizeValue(grayValue, 0, 255);
                    FlatedImg[i][j] = Neurons[0][index].input;
                }
            }
            return FlatedImg;
        }
        public void SetWeights()
        {
            int layers = Neurons.GetLength(0);
            Weights = new double[layers - 1][,];

            for (int l = 0; l < layers - 1; l++)
            {
                Weights[l] = new double[Neurons[l].Length, Neurons[l + 1].Length];
                for (int i = 0; i < Neurons[l].Length; i++)
                {
                    for (int j = 0; j < Neurons[l + 1].Length; j++)
                    {
                        Weights[l][i, j] = RandomizeWeights(l + 1);
                    }
                }
            }
        }
        public void SetBiases()
        {
            for (int l = 1; l < Neurons.Length; l++)
            {
                for (int n = 0; n < Neurons[l].Length; n++)
                {
                    Neurons[l][n].bias = 0.0;
                }
            }
        }
        public void Train(double[][] input, double[] y,double Alpha)
        {
            int LastLayer = Neurons.GetLength(0) - 1;
            int Iterations = 3;
            while (Iterations-- > 0)
            {
                for (int i = 0; i < input.Length; i++)
                {
                    ForwardPass(LastLayer,y);
                    Backpropagation();
                    for (int l = 1; l <= LastLayer; l++)
                    {
                        for (int n = 0; n < Neurons[l].Length; n++)
                        {
                            Neurons[l][n].bias -= Alpha * Neurons[l][n].error;
                            for (int j = 0; j < Neurons[l - 1].Length; j++)
                            {
                                Weights[l - 1][j, n] -= Alpha * Neurons[l - 1][j].output * Neurons[l][n].error;
                                if (L1) Weights[l - 1][j, n] -= Lambda * Math.Sign(Weights[l - 1][j, n]);
                                if (L2) Weights[l - 1][j, n] -= Lambda * Weights[l - 1][j, n];
                            }
                        }
                    }
                }
            }
        }
        public void ForwardPass(int LastLayer, double[] y)
        {
            double[] cost = new double[Neurons[LastLayer].Length];
            for (int l = 1; l < Neurons.Length; l++)
            {
                for (int i = 0; i < Neurons[l].Length; i++)
                {
                    double sum = 0;
                    for (int j = 0; j < Neurons[l - 1].Length; j++)
                    {
                        sum += Neurons[l - 1][j].input * Weights[l - 1][j, i];
                    }
                    Neurons[l][i].input = sum;
                    Neurons[l][i].output = Sigmoid(Neurons[l][i].input + Neurons[l][i].bias);
                }
            }
            for (int n = 0; n < Neurons[LastLayer].Length; n++)
            {
                cost[n] = Neurons[LastLayer][n].output - y[n];
                Neurons[LastLayer][n].error = cost[n] * SigmoidPrime(Neurons[LastLayer][n].input + Neurons[LastLayer][n].bias);
            }
            int outputlayer = Neurons.GetLength(0) - 1;
            double[] output = new double[Neurons[outputlayer].Length];
            double biggest = 0;
            int biggestindex = 0;
            for (int n = 0; n < output.Length; n++)
            {
                output[n] = Neurons[outputlayer][n].output;
                if (biggest < output[n])
                {
                    biggest = output[n];
                    biggestindex = n;
                }
            }
            if (y[biggestindex] == 1) good++;
            else bad++;
            acc = good / (bad + good);
            if ((good + bad) % 10000 == 0)
            {
                Console.WriteLine("Accuracy:" + acc * 100 + "%");
                Console.WriteLine("Good:" + good);
                Console.WriteLine("Bad:" + bad);
                Console.WriteLine(good + bad);
            }
        }
        public void TestModel(double[][] input, double[] y)
        {
            int LastLayer = Neurons.GetLength(0) - 1;
            for (int i = 0; i < input.GetLength(0); i++)
            {
                ForwardPass(LastLayer,y);    
            }
        }
        public void SetLayers()
        {
            Neurons[1] = new Neuron[32];
            Neurons[2] = new Neuron[10];
            for (int l = 1; l < Neurons.Length; l++)
            {
                for (int i = 0; i < Neurons[l].Length; i++)
                {
                    Neurons[l][i] = new Neuron();
                }
            }
        }
        public void Backpropagation()
        {
            for (int l = Neurons.Length - 2; l > 0; l--)
            {
                for (int i = 0; i < Neurons[l].Length; i++)
                {
                    double sum = 0.0;
                    for (int j = 0; j < Neurons[l + 1].Length; j++)
                    {
                        sum += (Weights[l][i, j] * Neurons[l + 1][j].error);
                    }
                    Neurons[l][i].error = sum * SigmoidPrime(Neurons[l][i].input + Neurons[l][i].bias);
                }
            }
        }
        public double Sigmoid(double x)
        {
            return 1 / (1 + Math.Exp(-x));
        }

        private double SigmoidPrime(double x)
        {
            return Sigmoid(x) * (1.0 - Sigmoid(x));
        }
        private double NormalizeValue(double value, double minValue, double maxValue)
        {
            return (value - minValue) / (maxValue - minValue);
        }
        private double RandomizeWeights(int layer)
        {
            int fanIn = (layer > 0) ? Neurons[layer - 1].Length : 0;
            double b = Math.Sqrt(1.0 / fanIn);
            return (Rnd.NextDouble() * 2 - 1) * b;
        }
        public double[] NumToArray(double num)
        {
            double[] array = new double[10];
            array[(int)num] = 1;
            return array;
        }
    }
}
