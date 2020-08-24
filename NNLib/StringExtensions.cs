using System;
using System.Globalization;
using System.IO;
using System.Text;

namespace NNLib
{
    public static class StringExtensions
    {
        public static double[] Deserialize(this string from)
        {
            var strReader = new StringReader(from);
            var strBuilder = new StringBuilder();

            string expected = "{LENGTH:";
            int r;
            for (int i = 0; i < expected.Length; i++)
                if ((char)(r = strReader.Read()) != expected[i])
                    throw new FormatException("Invalid format of xml ! Data representation has to start with {LENGTH}");


            while ((r = strReader.Read()) != '}' && r >= '0' && r <= '9')
                strBuilder.Append((char)r);

            int length;
            if (r == '}')
                length = int.Parse(strBuilder.ToString(), CultureInfo.InvariantCulture);
            else
                throw new FormatException("Invalid format of xml ! Invalid length representation !");

            var data = new double[length];
            int index = 0;
            strBuilder = new StringBuilder();

            while ((r = strReader.Read()) != -1)
            {
                if (r != ';')
                    strBuilder.Append((char)r);
                else
                {
                    data[index] = double.Parse(strBuilder.ToString(), CultureInfo.InvariantCulture);
                    index++;
                    strBuilder = new StringBuilder();
                }
            }

            data[index] = double.Parse(strBuilder.ToString(), CultureInfo.InvariantCulture);

            return data;
        }
    }
}
