package com.github.robertwsmith.ml_pipelines.cross_validation

import com.github.robertwsmith.ml_pipelines._
import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.SparkSession
import scopt.OptionParser

case class CrossValidationModelFitConfig(
  input: String = "examples/data/iris-train",
  crossValidation: String = "examples/model/cross_validation/",
  overwrite: Boolean = false
)

object ModelFit {

  val name: String = "cross-validation-model-fit"

  val parser: OptionParser[CrossValidationModelFitConfig] =
    new OptionParser[CrossValidationModelFitConfig](name) {
      head(name, "1.0")

      opt[String]("input")
        .action((x, c) => c.copy(input = x))
        .optional()

      opt[String]("cross_validation")
        .action((x, c) => c.copy(crossValidation = x))
        .optional()

      opt[Unit]("overwrite")
        .action((_, c) => c.copy(overwrite = true))
        .optional()
    }

  /** Machine Learning Pipeline Tutorial - Cross Validation Example
    */
  def main(args: Array[String]): Unit = {
    // Step #1 - Parse the command line arguments
    val parsed: CrossValidationModelFitConfig =
      parser.parse(args, CrossValidationModelFitConfig()) match {
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
    val randomForest = new RandomForestClassifier()
      .setLabelCol(stringIndexer.getOutputCol)
      .setFeaturesCol(vectorAssembler.getOutputCol)
      .setMaxBins(64)
      .setMaxDepth(4)
      .setMinInfoGain(0.01)
      .setNumTrees(64)

    // Step #9 - Create IndexToString instance
    val indexToString = new IndexToString()
      .setInputCol(randomForest.getPredictionCol)
      .setOutputCol(s"${randomForest.getPredictionCol}_species")
      .setLabels(stringIndexerModel.labels)

    val pipeline = new Pipeline().setStages(
      Array[PipelineStage](stringIndexerModel, vectorAssembler, randomForest, indexToString)
    )
    val paramGrid = new ParamGridBuilder()
      .addGrid(randomForest.maxDepth, Array(3, 4, 5, 6))
      .addGrid(randomForest.maxBins, Array(16, 32, 64))
      .addGrid(randomForest.numTrees, Array(32, 64, 128))
      .build()

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol(randomForest.getLabelCol)
      .setPredictionCol(randomForest.getPredictionCol)
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
      crossValidator.save(parsed.crossValidation)

    val randomForestClassificationModel = crossValidatorModel.bestModel
      .asInstanceOf[PipelineModel]
      .stages(2)
      .asInstanceOf[RandomForestClassificationModel]

    // Step #14 - Evaluate RandomForestClassificationModel metrics
    println(
      RandomForestMetricsReport(
        randomForestClassificationModel,
        vectorAssembler.getInputCols
      ).toString
    )

    // Step #15 - Stop Spark
    spark.stop()
  }

}
