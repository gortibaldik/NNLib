using System;
using System.Globalization;
using System.IO;
using System.Text;

namespace NNLib
{
    public static class StringExtensions
    {
        /// <summary>
        /// Reads the string of format "{LENGTH:x}d[0];d[1];..." and deserializes it 
        /// into the array of length x with elements d[0]...d[x-1]
        /// Throws exceptions if the string doesn't follow specified format, or if 
        /// any parsing fails.
        /// </summary>
        public static double[] DeserializeIntoDoubleArray(this string from)
        {
            var strReader = new StringReader(from);
            var strBuilder = new StringBuilder();

            // format checks
            string expected = "{LENGTH:";
            int r;
            for (int i = 0; i < expected.Length; i++)
                if ((char)(r = strReader.Read()) != expected[i])
                    throw new FormatException("Invalid format ! Data representation has to start with {LENGTH}");


            while ((r = strReader.Read()) != '}' && r >= '0' && r <= '9')
                strBuilder.Append((char)r);

            int length;
            if (r == '}')
                length = int.Parse(strBuilder.ToString(), CultureInfo.InvariantCulture);
            else
                throw new FormatException("Invalid format ! Invalid length representation !");


            // reading the actual data
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
