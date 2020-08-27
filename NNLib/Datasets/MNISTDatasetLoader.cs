using NNLib;
using System;
using System.IO;
using System.IO.Compression;
using System.Threading.Tasks;
using System.Collections.Generic;
using System.Net.Http;

namespace NNMnist
{
    class MNISTDatasetLoader
    {
        private static Random rng = new Random();

        private static string basePage = @"http://yann.lecun.com/exdb/mnist/";

        private static HttpClient client = new HttpClient() { Timeout = TimeSpan.FromMinutes(10) };


        private int nImages;

        private List<(double[] dataPoint, double[] label)> dataPoints;

        /// <summary>
        /// Loads the dataset. If the dataset isn't downloaded yet it downloads the dataset from the web and saves it to the 
        /// corresponding files. Then it prepares the data from the files to dataPoints usable with method GetBatch().
        /// </summary>
        public async Task LoadAsync(string datasetImages, string datasetLabels, int nImages)
        {
            if (!File.Exists(datasetImages) || !File.Exists(datasetLabels))
            {
                Func<string, Task> downloader = async (path) =>
                {
                    using Stream stream = await client.GetStreamAsync(basePage + path);
                    using var fileStream = File.OpenWrite(path);
                    await stream.CopyToAsync(fileStream);
                };

                await Task.WhenAll(
                    downloader(datasetImages),
                    downloader(datasetLabels));
            }

            
            using (var fileImages = new FileStream(datasetImages, FileMode.Open))
            using (var fileLabels = new FileStream(datasetLabels, FileMode.Open))
            using (var streamImages = new GZipStream(fileImages, CompressionMode.Decompress))
            using (var streamLabels = new GZipStream(fileLabels, CompressionMode.Decompress))
            {
                this.nImages = nImages;

                // not checking any of values
                // read 4 ints : magic, imageCount, rows, columns
                streamImages.Read(new byte[4 * 4], 0, 4 * 4);

                // read 2 ints : magic, labelCount
                streamLabels.Read(new byte[2 * 4], 0, 2 * 4);

                var dataImage = new byte[28 * 28];
                dataPoints = new List<(double[], double[])>();

                for (int i = 0; i < nImages; i++)
                {
                    // at first sight, there is possibility of asynchrony
                    // however streamImages.ReadAsync() shows a bug
                    var image = streamImages.Read(dataImage, 0, 28 * 28);
                    if (image != 28 * 28)
                        throw new DataMisalignedException("Images file corrupted");

                    var label = streamLabels.ReadByte();
                    byte dataLabel = label != -1 ? (byte)label : throw new DataMisalignedException("Label file corrupted!");

                    dataPoints.Add((BytesToDoubles(dataImage), IntToOneHot(label)));
                }
            }
        }

        /// <summary>
        /// Loads specified range of dataPoints and corresponding labels. fromIndex inclusive, toIndex exclusive
        /// </summary>
        public (Tensor dataPoints, Tensor labels) GetBatch(int fromIndex, int toIndex)
        {
            if (fromIndex < 0 || toIndex > nImages)
                throw new ArgumentOutOfRangeException();

            var ps = new double[toIndex - fromIndex][];
            var ls = new double[toIndex - fromIndex][];
            for (int i = fromIndex; i < toIndex; i++)
            {
                ps[i - fromIndex] = dataPoints[i].dataPoint;
                ls[i - fromIndex] = dataPoints[i].label;
            }

            return (new Tensor(1, 28, 28, ps), new Tensor(1, 10, 1, ls));
        }


        private double[] BytesToDoubles(byte[] data)
        {
            double[] result = new double[data.Length];
            for (int i = 0; i < data.Length; i++)
            {
                result[i] = data[i] > 100 ? 1 : 0;
            }

            return result;
        }

        private double[] IntToOneHot(int i)
        {
            if (i < 0 || i > 9)
                throw new ArgumentOutOfRangeException($"{nameof(i)} should be between 0 and 9!");

            double[] data = new double[10];
            data[i] = 1D;
            return data;
        }

        /// <summary>
        /// Shuffles the dataPoints. 
        /// </summary>
        public void Shuffle()
        {
            int n = dataPoints.Count;
            while (n > 1)
            {
                n--;
                var k = rng.Next(n + 1);
                var value = dataPoints[k];
                dataPoints[k] = dataPoints[n];
                dataPoints[n] = value;
            }
        }
    }
}
