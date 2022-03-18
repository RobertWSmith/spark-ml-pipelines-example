package com.github.robertwsmith.ml_pipelines.xgboost_pipeline

import com.github.robertwsmith.ml_pipelines.{makeSaveMode, ModelFitMetrics}
import com.github.robertwsmith.ml_pipelines.pipeline.PipelinePredictConfig
import org.apache.spark.ml.classification.RandomForestClassificationModel
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.SparkSession

object Predict {
  val name: String = "xgboost-pipeline-predict"

  /** Pipeline Model Predict
    *
    * @param args command line arguments
    */
  def main(args: Array[String]): Unit = {
    // Step 1
    val parsed: PipelinePredictConfig =
      com.github.robertwsmith.ml_pipelines.pipeline.Predict.parser
        .parse(args.toSeq, PipelinePredictConfig()) match {
        case Some(c) => c
        case None =>
          throw new Exception(s"Malformed command line arguments: ${args.mkString(", ")}")
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
    val labelCol      = randomForest.getOrDefault(randomForest.labelCol)
    val predictionCol = randomForest.getOrDefault(randomForest.predictionCol)

    println(ModelFitMetrics(prediction, labelCol, predictionCol).toString)

    // Step 8
    spark.stop()
  }

}
