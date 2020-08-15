using System;
using System.Collections.Generic;
using System.Text;

namespace NNLib
{
    public interface IDataset : IEnumerable<(Tensor Input, Tensor Label)>
    {
        int Epochs { get; set; }
        int BatchSize { get; set; }
        bool EndTraining { get; }
        bool EndEpoch { get; }
        IEnumerable<(Tensor Input, Tensor Label)> GetBatch();

        IEnumerable<(Tensor Input, Tensor Label)> GetValidation();

        IEnumerable<(Tensor Input, Tensor Label)> GetTestSet();
    }
}
