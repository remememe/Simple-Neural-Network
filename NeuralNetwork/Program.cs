using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
namespace NeuralNetwork
{
    internal class Program
    {
        static void Main()
        {
            Forward forward = new Forward();
            Program program = new Program();

            Console.WriteLine("Images Path:");
            string ImgPath = Console.ReadLine();
            LoadImages.LoadImg(forward,ImgPath);

            Console.WriteLine("Read Data? Y/N");
            string ReadData = Console.ReadLine();
            forward.SetLayers();
            if (ReadData == "N")
            {
                forward.SetBiases();
                forward.SetWeights();
            }
            else
            {
                Console.WriteLine("Data Path:");
                string DataPath = Console.ReadLine();
                SaveData.ImportBias(DataPath + "Biases.csv", forward.Neurons);
                forward.Weights = SaveData.ImportWeights(DataPath + "weights.csv");
            }
            Console.WriteLine("Train Model Y/N");
            string Train = Console.ReadLine();

            if (Train == "Y") program.TrainMain(forward);
            else
            {
                program.TestModel(forward);
            }
            Console.ReadLine();
        }
        public void TrainMain(Forward forward)
        {
            for (int i = 0; i < LoadImages.TrainingImages.Length; i++)
            {
                for (int j = 0; j < LoadImages.TrainingImages[i].Length; j++)
                {
                    double[][] Input = LoadImages.TrainingImages[i][j];
                    double[] Exp = forward.NumToArray(LoadImages.trainingOutputs[i][j]);
                    forward.Train(Input, Exp, 3f);
                }
            }
            Console.WriteLine("Accuracy:" + forward.acc * 100 + "%");
            Console.WriteLine("Good:" + forward.good);
            Console.WriteLine("Bad:" + forward.bad);
            Console.WriteLine("Save Data? Y/N");
            string answer = Console.ReadLine();
            if (answer == "Y" || answer == "y")
            {
                Console.WriteLine("Data Path:");
                string DataPath = Console.ReadLine();
                SaveData.ExportWeights(forward.Weights,DataPath + "weights.csv");
                SaveData.ExportBias(forward.Neurons,DataPath + "Biases.csv");
            }
        }
        public void TestModel(Forward forward)
        {
            for (int i = 0; i < LoadImages.TrainingImages.Length; i++)
            {
                for (int j = 0; j < LoadImages.TrainingImages[i].Length; j++)
                {
                    double[][] Input = LoadImages.TrainingImages[i][j];
                    double[] Exp = forward.NumToArray(LoadImages.trainingOutputs[i][j]);
                    forward.ForwardPass(2, Exp);
                }
            }
            Console.WriteLine("Accuracy:" + forward.acc * 100 + "%");
            Console.WriteLine("Good:" + forward.good);
            Console.WriteLine("Bad:" + forward.bad);
        }
    }
}
