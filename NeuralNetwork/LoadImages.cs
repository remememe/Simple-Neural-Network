using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    internal class LoadImages
    {
        public static double[][] trainingOutputs;
        public static double[][][][] TrainingImages;
        public static void LoadImg(Forward forward,string source)
        {   
            Bitmap[][] trainingInputs = LoadTrainingInputs(ref trainingOutputs,source);
            TrainingImages = new double[trainingInputs.Length][][][];
            for (int i = 0; i < trainingInputs.Length; i++)
            {
                TrainingImages[i] = new double[trainingInputs[i].Length][][];

                for (int j = 0; j < trainingInputs[i].Length; j++)
                {
                    TrainingImages[i][j] = forward.FlatImage(ref trainingInputs[i][j]);
                }
            }
        }
        static Bitmap[][] LoadTrainingInputs(ref double[][] trainingOutputs, string source)
        {
            Bitmap[][] Images = new Bitmap[10][];
            trainingOutputs = new double[10][];
            string Path = source;
            for (int i = 0; i < 10; i++)
            {
                string[] files = Directory.GetFiles(Path + @"\" + i, "*.png");
                Images[i] = new Bitmap[files.Length / 100];
                trainingOutputs[i] = new double[files.Length / 100];
                for (int j = 0; j < files.Length / 100; j++)
                {
                    Bitmap bitmap = new Bitmap(files[j]);
                    Images[i][j] = bitmap;
                    trainingOutputs[i][j] = i;
                }
            }
            Console.WriteLine("Loaded");
            return Images;
        }
    }
}
