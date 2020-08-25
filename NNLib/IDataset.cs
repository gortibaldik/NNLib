using System;
using System.Collections.Generic;
using System.Text;

namespace NNLib
{
    public interface IDataset
    {
        
        int Epochs { get; set; }

        /// <summary>
        /// The size of batch of the train part of the dataset.
        /// </summary>
        int BatchSize { get; set; }

        /// <summary>
        /// The dataset should set this flag on after ending last epoch of training
        /// </summary>
        bool EndTraining { get; }

        /// <summary>
        /// The dataset should set this flag on after end of each epoch
        /// </summary>
        bool EndEpoch { get; }

        /// <summary>
        /// The dataset should return tensor with the BatchSize layers of dataPoints and
        /// another tensor with the BatchSize layers of Labels
        /// </summary>
        (Tensor Input, Tensor Label) GetBatch();

        /// <summary>
        /// The dataset should return one big batch of validation data and validation labels
        /// </summary>
        (Tensor Input, Tensor Label) GetValidation();

        /// <summary>
        /// The dataset should return one big batch of test data and test labels
        /// </summary>
        (Tensor Input, Tensor Label) GetTestSet();
    }
}
