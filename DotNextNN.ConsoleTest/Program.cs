using CLAP;

namespace DotNextNN.ConsoleTest
{
    class Program
    {
        static void Main(string[] args)
        {
            Parser.Run(args, new ConsoleTest.App());
        }
    }
}
