using System;
using System.IO;
using System.IO.Compression;
using System.Reflection;
using CLAP;

namespace DotNextNN.ConsoleTest
{
    class Program
    {
        static void Main(string[] args)
        {
            if (IsWindows())
            {
                UnpackOpenBlas();
            }
            
            Parser.Run(args, new ConsoleTest.App());
        }

        private static bool IsWindows()
        {
            string windir = Environment.GetEnvironmentVariable("windir");
            return (!string.IsNullOrEmpty(windir) && windir.Contains(@"\") && Directory.Exists(windir));
        }

        private static void UnpackOpenBlas()
        {
            var assembly = Assembly.GetExecutingAssembly();
            string dir = Path.GetDirectoryName(assembly.Location);
            if (File.Exists(Path.Combine(dir, "libopenblas.dll")))
            {
                return;
            }
            
            var resource = assembly.GetManifestResourceStream($"{assembly.GetName().Name}.libopenblas.zip");
            if (resource == null)
            {
                throw new InvalidOperationException("Resource not available.");
            }

            var zip = new ZipArchive(resource, ZipArchiveMode.Read);
            zip.ExtractToDirectory(dir);
        }
    }
}
