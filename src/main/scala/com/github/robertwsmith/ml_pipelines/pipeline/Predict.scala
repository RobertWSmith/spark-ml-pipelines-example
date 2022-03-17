package com.github.robertwsmith.ml_pipelines.pipeline

import com.github.robertwsmith.ml_pipelines.{
  makeSaveMode,
  modelFitMetrics,
  ModelFitMetrics
}
import org.apache.spark.ml.classification.RandomForestClassificationModel
import org.apache.spark.ml.feature.{
  IndexToString,
  StringIndexerModel,
  VectorAssembler
}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.SparkSession
import scopt.OptionParser

case class PipelinePredictConfig(
    input: String = "examples/data/iris-test",
    output: String = "examples/data/pipeline/iris-test-predict",
    pipeline: String = "examples/model/pipeline/",
    overwrite: Boolean = false
)

class Predict {
  val name: String = "pipeline-predict"

  val parser = new OptionParser[PipelinePredictConfig](name) {
    head(name, "1.0")

    opt[String]("input")
      .action((x, c) => c.copy(input = x))
      .optional()

    opt[String]("output")
      .action((x, c) => c.copy(output = x))
      .optional()

    opt[String]("pipeline")
      .action((x, c) => c.copy(pipeline = x))
      .optional()

    opt[Unit]("overwrite")
      .action((_, c) => c.copy(overwrite = true))
      .optional()
  }

  /** Pipeline Model Predict
    *
    * @param args command line arguments
    */
  def main(args: Array[String]): Unit = {
    // Step 1
    val parsed: PipelinePredictConfig =
      this.parser.parse(args, PipelinePredictConfig()) match {
        case Some(c) => c
        case None =>
          throw new Exception(s"Malformed command line arguments: ${args}")
      }

    // Step 2
    val spark = SparkSession.builder().getOrCreate()

    val pipelineModel = PipelineModel.load(parsed.pipeline)

    val testDF = spark.read.parquet(parsed.input)

    val prediction = pipelineModel.transform(testDF).repartition(1)

    prediction.write.mode(makeSaveMode(parsed.overwrite)).parquet(parsed.output)

    val randomForest = pipelineModel
      .stages(2)
      .asInstanceOf[RandomForestClassificationModel]
    val labelCol = randomForest.getOrDefault(randomForest.labelCol)
    val predictionCol = randomForest.getOrDefault(randomForest.predictionCol)

    println(ModelFitMetrics(prediction, labelCol, predictionCol).toString)

    // Step 8
    spark.stop()
  }

}
