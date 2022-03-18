package com.github.robertwsmith.ml_pipelines.xgboost_cross_validation

import com.github.robertwsmith.ml_pipelines.{makeSaveMode, ModelFitMetrics}
import com.github.robertwsmith.ml_pipelines.cross_validation.CrossValidationPredictConfig
import ml.dmlc.xgboost4j.scala.spark.XGBoostClassificationModel
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.tuning.CrossValidatorModel
import org.apache.spark.sql.SparkSession

object Predict {
  val name: String = "xgboost-pipeline-predict"

  /** Pipeline Model Predict
    *
    * @param args command line arguments
    */
  def main(args: Array[String]): Unit = {
    // Step 1
    val parsed: CrossValidationPredictConfig =
      com.github.robertwsmith.ml_pipelines.cross_validation.Predict.parser
        .parse(args.toSeq, CrossValidationPredictConfig()) match {
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

    val xgboostModel = crossValidatorModel.bestModel
      .asInstanceOf[PipelineModel]
      .stages(2)
      .asInstanceOf[XGBoostClassificationModel]
    val labelCol      = xgboostModel.getOrDefault(xgboostModel.labelCol)
    val predictionCol = xgboostModel.getOrDefault(xgboostModel.predictionCol)

    println(ModelFitMetrics(prediction, labelCol, predictionCol).toString)

    // Step 8
    spark.stop()
  }

}
