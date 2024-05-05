using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    internal class SaveData
    {
        public static void ExportWeights(double[][,] weights, string filePath)
        {
            using (StreamWriter writer = new StreamWriter(filePath))
            {
                foreach (var layerWeights in weights)
                {
                    int numRows = layerWeights.GetLength(0);
                    int numCols = layerWeights.GetLength(1);

                    for (int i = 0; i < numRows; i++)
                    {
                        for (int j = 0; j < numCols; j++)
                        {
                            writer.Write(layerWeights[i, j]);

                            if (j < numCols - 1)
                                writer.Write("|");
                        }
                        writer.WriteLine();
                    }
                    writer.WriteLine();
                }
            }
        }
        public static void ExportBias(Neuron[][] Neuron, string filePath)
        {
            using (StreamWriter writer = new StreamWriter(filePath))
            {
                for (int i = 0; i < Neuron.Length; i++)
                {
                    for (int j = 0; j < Neuron[i].Length; j++)
                    {
                        writer.Write(Neuron[i][j].bias);
                        if(j < Neuron[i].Length - 1)
                        {
                            writer.Write("|");
                        }
                    }
                    writer.WriteLine();
                }
                writer.WriteLine();
            }
        }
        public static double[][,] ImportWeights(string filePath)
        {
            List<double[,]> weightsList = new List<double[,]>();
            using (StreamReader reader = new StreamReader(filePath))
            {
                List<List<double>> layerWeights = new List<List<double>>();
                string line;
                while ((line = reader.ReadLine()) != null)
                {
                    if (line.Trim() == "")
                    {
                        double[,] layerWeightsArray = new double[layerWeights.Count, layerWeights[0].Count];
                        for (int i = 0; i < layerWeights.Count; i++)
                        {
                            for (int j = 0; j < layerWeights[0].Count; j++)
                            {
                                layerWeightsArray[i, j] = layerWeights[i][j];
                            }
                        }
                        weightsList.Add(layerWeightsArray);
                        layerWeights.Clear();
                        continue;
                    }
                    string[] values = line.Split('|');
                    List<double> weights = new List<double>();
                    foreach (string value in values)
                    {
                        weights.Add(double.Parse(value));
                    }
                    layerWeights.Add(weights);
                    Console.WriteLine(layerWeights.Count);
                }
            }
            return weightsList.ToArray();
        }
        public static void ImportBias(string filePath, Neuron[][] Neurons)
        {
            using (StreamReader reader = new StreamReader(filePath))
            {
                int layerIndex = 0;
                string line;
                while ((line = reader.ReadLine()) != null)
                {
                    if (line.Trim() == "")
                    {
                        layerIndex++;
                        continue;
                    }
                    string[] values = line.Split('|');
                    for (int i = 0; i < values.Length; i++)
                    {
                        Neurons[layerIndex][i].bias = double.Parse(values[i]);
                    }
                }
            }
        }
    }
}
