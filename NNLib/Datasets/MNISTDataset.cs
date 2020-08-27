using System;
using System.Threading.Tasks;
using NNLib.Datasets;
using NNLib;

namespace NNMnist
{
    class MNISTDataset : IDataset
    {
        private MNISTDatasetLoader trainLoader;
        private MNISTDatasetLoader testLoader;

        private bool loadedTraining;
        private bool loadedTest;

        private int cImage;

        private string trainDatasetImages = "train-images-idx3-ubyte.gz";
        private string trainDatasetLabels = "train-labels-idx1-ubyte.gz";
        private int trainImages = 50_000;
        private int validationImages = 10_000;

        private string testDatasetImages = "t10k-images-idx3-ubyte.gz";
        private string testDatasetLabels = "t10k-labels-idx1-ubyte.gz";
        private int testImages = 10_000;

        public bool EndTraining { get; private set; }
        public bool EndEpoch { get; private set; }
        public int Epochs { get; set; }

        private int? batchSize = null;

        /// <summary>
        /// Adjustable parameter, specifies the BatchSize dimension of the tensor returned from the method GetBatch()
        /// </summary>
        public int BatchSize { 
            get => batchSize == null ? 32 : batchSize.Value;
            set
            {
                batchSize = cImage != 0 ? throw new InvalidOperationException() : value;
                if (batchSize < 0)
                    throw new ArgumentOutOfRangeException();
            }
        }

        /// <summary>
        /// Initializes the class
        /// </summary>
        public MNISTDataset()
        {
            this.trainLoader = new MNISTDatasetLoader();
        }

        /// <summary>
        /// Loads the training dataset. If the training dataset isn't downloaded yet, it downloads the files and then prepares the data for the training.
        /// </summary>
        public async Task LoadTrainSetAsync()
        {
            await trainLoader.LoadAsync(trainDatasetImages, trainDatasetLabels, trainImages + validationImages);
            trainLoader.Shuffle();
            loadedTraining = true;

            cImage = 0;
        }

        /// <summary>
        /// Loads the test dataset. If the training dataset isn't downloaded yet, it downloads the files and then prepares the data for the training.
        /// </summary>
        public async Task LoadTestSetAsync()
        {
            testLoader = new MNISTDatasetLoader();
            loadedTest = true;
            await testLoader.LoadAsync(testDatasetImages, testDatasetLabels, testImages);
        }

        /// <summary>
        /// Gets BatchSize images and labels from the training dataset and organizes them to Input and Label tensors.
        /// May set the flags EndEpoch and EndTraining.
        /// Throws InvalidOperationException if the training dataset hasn't been downloaded yet via LoadTrainSetAsync()
        /// </summary>
        public (Tensor Input, Tensor Label) GetBatch()
        {
            if (!loadedTraining)
                throw new InvalidOperationException("Training dataset wasn't downloaded yet !");

            var from = cImage;
            var to = cImage + BatchSize;
            cImage += BatchSize;

            if (to > trainImages)
            {
                to = trainImages;
                EndEpoch = true;
                Epochs--;
                cImage = 0;
                if (to <= 0)
                    return (null, null);
            }
            if (Epochs == 0)
                EndTraining = true;

            return trainLoader.GetBatch(from, to);
        }

        /// <summary>
        /// Returns Input and Label tensors containing one big batch of all the validation data.
        /// Throws InvalidOperationException if the training dataset hasn't been downloaded yet via LoadTrainSetAsync()
        /// </summary>
        public (Tensor Input, Tensor Label) GetValidation()
        {
            if (!loadedTraining)
                throw new InvalidOperationException("Training dataset wasn't downloaded yet !");

            EndEpoch = false;
            return trainLoader.GetBatch(50_000, 60_000);
        }

        /// <summary>
        /// Returns Input and Label tensors containing one big batch of all the test data.
        /// Throws InvalidOperationException if the test dataset hasn't been downloaded yet via LoadTrainSetAsync()
        /// </summary>
        public (Tensor Input, Tensor Label) GetTestSet()
        {
            if (!loadedTest)
                throw new InvalidOperationException();

            return testLoader.GetBatch(0, testImages);
        }

    }
}
