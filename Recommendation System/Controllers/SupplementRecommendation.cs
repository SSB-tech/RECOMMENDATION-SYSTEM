using Microsoft.AspNetCore.Mvc;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace SupplementRecommendation.Controllers
{
    [ApiController]
    public class SupplementRecommendationController : ControllerBase
    {
        private readonly MLContext mlContext;

        public SupplementRecommendationController()
        {
            mlContext = new MLContext();
        }

        [HttpGet]
        [Route("api/recommendations")]
        public IActionResult GetRecommendation(float HealthCondition, float FitnessGoal)
        {
            // Load data
            var dataView = mlContext.Data.LoadFromTextFile<SupplementData>(
                path: "data/dataset.csv",
                hasHeader: true,
                separatorChar: ',',
                allowQuoting: true,
                allowSparse: false);

            // Define pipeline
            var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label", "Recommendation")
                .Append(mlContext.Transforms.Concatenate("Features", "HealthCondition", "FitnessGoal"))
                .Append(mlContext.Transforms.NormalizeMinMax("Features"))
                .Append(mlContext.MulticlassClassification.Trainers.SdcaNonCalibrated())
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            // Train model
            var model = pipeline.Fit(dataView);

            // Make prediction
            var predictionEngine = mlContext.Model.CreatePredictionEngine<SupplementData, SupplementPrediction>(model);
            var prediction = predictionEngine.Predict(new SupplementData { HealthCondition = HealthCondition, FitnessGoal = FitnessGoal });

            // Return recommendation
            return Ok(prediction.Supplement);
        }

        public class SupplementData
        {
            [LoadColumn(0)]
            public float HealthCondition { get; set; }

            [LoadColumn(1)]
            public float FitnessGoal { get; set; }

            [LoadColumn(2)]
            public string Supplement { get; set; }

            [LoadColumn(3)]
            public string Recommendation { get; set; }
        }

        public class SupplementPrediction
        {
            [ColumnName("PredictedLabel")]
            public string Supplement { get; set; }

            [ColumnName("Score")]
            public float[] Scores { get; set; }
        }
    }
}

//==================================================================================================

//using Microsoft.AspNetCore.Mvc;
//using Microsoft.ML;
//using Microsoft.ML.Data;

//namespace SupplementRecommendation.Controllers
//{
//    [ApiController]
//    public class SupplementRecommendationController : ControllerBase
//    {
//        private readonly MLContext mlContext;

//        public SupplementRecommendationController()
//        {
//            mlContext = new MLContext();
//        }

//        [HttpGet]
//        [Route("api/recommendations")]
//        public IActionResult GetRecommendation(float bmi, float bmr)
//        {
//            // Load data
//            var dataView = mlContext.Data.LoadFromTextFile<SupplementData>(
//                path: "data/test2.csv",
//                hasHeader: true,
//                separatorChar: ',',
//                allowQuoting: true,
//                allowSparse: false);

//            // Define pipeline
//            var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label", "Recommendation")
//                .Append(mlContext.Transforms.Concatenate("Features", "Bmi", "Bmr"))
//                .Append(mlContext.Transforms.NormalizeMinMax("Features"))
//                .Append(mlContext.MulticlassClassification.Trainers.SdcaNonCalibrated())
//                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

//            // Train model
//            var model = pipeline.Fit(dataView);

//            // Make prediction
//            var predictionEngine = mlContext.Model.CreatePredictionEngine<SupplementData, SupplementPrediction>(model);
//            var prediction = predictionEngine.Predict(new SupplementData { Bmi = bmi, Bmr = bmr });

//            // Return recommendation
//            return Ok(prediction.Supplement);
//        }

//        public class SupplementData
//        {
//            [LoadColumn(0)]
//            public float Bmi { get; set; }

//            [LoadColumn(1)]
//            public float Bmr { get; set; }

//            [LoadColumn(2)]
//            public string Supplement { get; set; }

//            [LoadColumn(3)]
//            public string Recommendation { get; set; }
//        }

//        public class SupplementPrediction
//        {
//            [ColumnName("PredictedLabel")]
//            public string Supplement { get; set; }

//            [ColumnName("Score")]
//            public float[] Scores { get; set; }
//        }
//    }
//}








