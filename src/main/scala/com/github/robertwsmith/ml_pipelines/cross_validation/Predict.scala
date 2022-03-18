package com.github.robertwsmith.ml_pipelines.cross_validation

import com.github.robertwsmith.ml_pipelines.{makeSaveMode, ModelFitMetrics}
import org.apache.spark.ml.classification.RandomForestClassificationModel
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.tuning.CrossValidatorModel
import org.apache.spark.sql.SparkSession
import scopt.OptionParser

case class CrossValidationPredictConfig(
  input: String = "examples/data/iris-train",
  output: String = "examples/data/cross_validation/iris-test-predict",
  crossValidation: String = "examples/model/cross_validation/",
  overwrite: Boolean = false
)

object Predict {
  val name: String = "cross-validation-predict"

  val parser: OptionParser[CrossValidationPredictConfig] =
    new OptionParser[CrossValidationPredictConfig](name) {
      head(name, "1.0")

      opt[String]("input")
        .action((x, c) => c.copy(input = x))
        .optional()

      opt[String]("output")
        .action((x, c) => c.copy(output = x))
        .optional()

      opt[String]("cross_validation")
        .action((x, c) => c.copy(crossValidation = x))
        .optional()

      opt[Unit]("overwrite")
        .action((_, c) => c.copy(overwrite = true))
        .optional()
    }

  /** Cross Validation Model Predict
    *
    * @param args command line arguments
    */
  def main(args: Array[String]): Unit = {
    // Step 1
    val parsed: CrossValidationPredictConfig =
      this.parser.parse(args, CrossValidationPredictConfig()) match {
        case Some(c) => c
        case None =>
          throw new Exception(s"Malformed command line arguments: ${args.mkString(", ")}")
      }

    // Step 2
    val spark = SparkSession.builder().getOrCreate()

    val crossValidatorModel = CrossValidatorModel.load(parsed.crossValidation)

    val testDF = spark.read.parquet(parsed.input)

    val prediction = crossValidatorModel.transform(testDF).repartition(1)

    prediction.write.mode(makeSaveMode(parsed.overwrite)).parquet(parsed.output)

    val randomForest = crossValidatorModel.bestModel
      .asInstanceOf[PipelineModel]
      .stages(2)
      .asInstanceOf[RandomForestClassificationModel]
    val labelCol      = randomForest.getOrDefault(randomForest.labelCol)
    val predictionCol = randomForest.getOrDefault(randomForest.predictionCol)

    println(ModelFitMetrics(prediction, labelCol, predictionCol).toString)

    // Step 8
    spark.stop()
  }

}
