using System;
using System.Collections.Generic;
using System.Text;

namespace NNLib
{
    public interface IDataset
    {
        int Epochs { get; set; }
        int BatchSize { get; set; }
        bool EndTraining { get; }
        bool EndEpoch { get; }
        (Tensor Input, Tensor Label) GetBatch();

        (Tensor Input, Tensor Label) GetValidation();

        (Tensor Input, Tensor Label) GetTestSet();
    }
}
