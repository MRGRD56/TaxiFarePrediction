using System;
using System.IO;
using Microsoft.ML;
using TaxiFarePrediction.Models;

namespace TaxiFarePrediction
{
    internal static class Program
    {
        private static readonly string _trainDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-train.csv");
        private static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-test.csv");
        private static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");
        
        private static void Main(string[] args)
        {
            var mlContext = new MLContext(0);

            Console.WriteLine("Обучение модели...");
            var model = Train(mlContext, _trainDataPath);
            Console.WriteLine("Модель обучена.");
            
            Evaluate(mlContext, model);
            
            TestSinglePrediction(mlContext, model);
        }

        private static ITransformer Train(MLContext mlContext, string dataPath)
        {
            var dataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(dataPath, ',', true);

            // var pipeline = mlContext.Transforms.CopyColumns("Label", "FareAmount")
            //     .Append(mlContext.Transforms.Categorical.OneHotEncoding("VendorIdEncoded", "VendorId"))
            //     .Append(mlContext.Transforms.Categorical.OneHotEncoding("RateCodeEncoded", "RateCode"))
            //     .Append(mlContext.Transforms.Categorical.OneHotEncoding("PaymentTypeEncoded", "PaymentType"))
            //     .Append(mlContext.Transforms.Concatenate("Features",
            //         "VendorIdEncoded", "RateCodeEncoded", "PassengerCount",
            //         "TripTime", "TripDistance", "PaymentTypeEncoded", "FareAmount"))
            //     .Append(mlContext.Regression.Trainers.FastTree());
            
            var pipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "FareAmount")
                // </Snippet7>
                // <Snippet8>
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "VendorIdEncoded", inputColumnName: "VendorId"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "RateCodeEncoded", inputColumnName: "RateCode"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "PaymentTypeEncoded", inputColumnName: "PaymentType"))
                // </Snippet8>
                // <Snippet9>
                .Append(mlContext.Transforms.Concatenate("Features", "VendorIdEncoded", "RateCodeEncoded", "PassengerCount", "TripDistance", "PaymentTypeEncoded"))
                // </Snippet9>
                // <Snippet10>
                .Append(mlContext.Regression.Trainers.FastTree());

            return pipeline.Fit(dataView);
        }

        private static void Evaluate(MLContext mlContext, ITransformer model)
        {
            var dataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(_testDataPath, ',', true);
            var predictions = model.Transform(dataView);
            var metrics = mlContext.Regression.Evaluate(predictions);
            
            Console.WriteLine(@"
*************************************************
*       Model quality metrics evaluation         
*------------------------------------------------");
            
            Console.WriteLine($"*       RSquared Score:      {metrics.RSquared:N2}");
            Console.WriteLine($"*       Root Mean Squared Error:      {metrics.RootMeanSquaredError:N2}");
        }

        private static void TestSinglePrediction(MLContext mlContext, ITransformer model)
        {
            var predicationEngine = mlContext.Model.CreatePredictionEngine<TaxiTrip, TaxiTripFarePrediction>(model);
            
            var taxiTripSample = new TaxiTrip
            {
                VendorId = "VTS",
                RateCode = "1",
                PassengerCount = 1,
                TripTime = 1140,
                TripDistance = 3.75f,
                PaymentType = "CRD",
                FareAmount = 0 // To predict. Actual/Observed = 15.5
            };

            // var taxiTripSample = new TaxiTrip
            // {   //CMT,1,1,339,0.8,CRD,5.5
            //     VendorId = "CMT",
            //     RateCode = "1",
            //     PassengerCount = 1,
            //     TripTime = 339,
            //     TripDistance = 0.8f,
            //     PaymentType = "CRD",
            //     FareAmount = 0 //5.5
            // };

            var prediction = predicationEngine.Predict(taxiTripSample);
            
            Console.WriteLine($"**********************************************************************");
            Console.WriteLine($"Predicted fare: {prediction.FareAmount:0.####}, actual fare: {15.5}");
            Console.WriteLine($"**********************************************************************");
        }
    }
}