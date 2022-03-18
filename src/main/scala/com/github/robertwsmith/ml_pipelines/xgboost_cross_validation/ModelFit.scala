package com.github.robertwsmith.ml_pipelines.xgboost_cross_validation

import com.github.robertwsmith.ml_pipelines._
import com.github.robertwsmith.ml_pipelines.cross_validation.{
  CrossValidationModelFitConfig,
  ModelFit => CVModelFit
}
import ml.dmlc.xgboost4j.scala.spark.{XGBoostClassificationModel, XGBoostClassifier}
import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.SparkSession

object ModelFit {

  def main(args: Array[String]): Unit = {
    // Step #1 - Parse the command line arguments
    val parsed: CrossValidationModelFitConfig =
      CVModelFit.parser.parse(args.toSeq, CrossValidationModelFitConfig()) match {
        case Some(c) => c
        case None =>
          throw new Exception(s"Malformed command line arguments: ${args.mkString(", ")}")
      }

    // Step #2 - Start SparkSession
    val spark = SparkSession.builder().getOrCreate()

    // Step #3 - Read in the training DataFrame
    val trainDF = spark.read.parquet(parsed.input)

    // Step #4 - Create StringIndexer instance, then fit a StringIndexerModel
    val stringIndexer = new StringIndexer()
      .setInputCol(targetColumnName)
      .setOutputCol(labelColumnName)
    val stringIndexerModel = stringIndexer.fit(trainDF)

    // Step #6 - Create VectorAssembler Instance
    val vectorAssembler = new VectorAssembler()
      .setInputCols(predictorColumnNames)
      .setOutputCol(featureColumnName)

    // Step #8 - Create a RandomForestClassifier instance
    val xgboostClassifier = new XGBoostClassifier()
      .setLabelCol(stringIndexer.getOutputCol)
      .setFeaturesCol(vectorAssembler.getOutputCol)
      .setObjective("multi:softprob")
      .setEta(0.3)
      .setNumClass(3)
      .setMaxDepth(3)
      .setMaxBins(32)
      .setNumRound(100)
      .setNumWorkers(1)
      .setNthread(1)

    // Step #9 - Create IndexToString instance
    val indexToString = new IndexToString()
      .setInputCol(xgboostClassifier.getPredictionCol)
      .setOutputCol(s"${xgboostClassifier.getPredictionCol}_species")
      .setLabels(stringIndexerModel.labels)

    val pipeline = new Pipeline().setStages(
      Array[PipelineStage](stringIndexerModel, vectorAssembler, xgboostClassifier, indexToString)
    )

    val paramGrid = new ParamGridBuilder()
      .addGrid(xgboostClassifier.maxDepth, Array(2, 3, 4, 5, 6))
      .addGrid(xgboostClassifier.maxBins, Array(16, 32, 64))
      .addGrid(xgboostClassifier.eta, Array(0.1, 0.2, 0.3, 0.4))
      .build()

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol(xgboostClassifier.getLabelCol)
      .setPredictionCol(xgboostClassifier.getPredictionCol)
      .setMetricName("accuracy")

    val crossValidator = new CrossValidator()
      .setEstimator(pipeline)
      .setEstimatorParamMaps(paramGrid)
      .setEvaluator(evaluator)
      .setNumFolds(3)
      .setParallelism(4)

    val crossValidatorModel = crossValidator.fit(trainDF)

    if (parsed.overwrite)
      crossValidatorModel.write.overwrite().save(parsed.crossValidation)
    else
      crossValidatorModel.save(parsed.crossValidation)

    val xgboostClassificationModel =
      crossValidatorModel.bestModel
        .asInstanceOf[PipelineModel]
        .stages(2)
        .asInstanceOf[XGBoostClassificationModel]

    // Step #14 - Evaluate RandomForestClassificationModel metrics
    println(XGBoostMetricsReport(xgboostClassificationModel, vectorAssembler.getInputCols).toString)

    // Step #15 - Stop Spark
    spark.stop()
  }

}
