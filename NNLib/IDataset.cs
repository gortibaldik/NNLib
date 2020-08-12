using System;
using System.Collections.Generic;
using System.Text;

namespace NNLib
{
    public interface IDataset : IEnumerable<(Tensor, Tensor)>
    {
        bool EndTraining { get; }
        IEnumerable<(Tensor Input, Tensor Label)> GetBatch();

        (Tensor Input, Tensor Label) GetValidation();
    }
}
