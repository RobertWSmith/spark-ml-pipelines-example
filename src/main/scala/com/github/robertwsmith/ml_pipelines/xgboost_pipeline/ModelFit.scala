package com.github.robertwsmith.ml_pipelines.xgboost_pipeline

import ml.dmlc.xgboost4j.scala.spark.{
  XGBoostClassificationModel,
  XGBoostClassifier
}
import com.github.robertwsmith.ml_pipelines._
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.classification.{
  RandomForestClassificationModel,
  RandomForestClassifier
}
import org.apache.spark.ml.feature.{
  IndexToString,
  StringIndexer,
  VectorAssembler
}
import org.apache.spark.ml.util.MLWritable
import org.apache.spark.sql.SparkSession
import scopt.OptionParser
import com.github.robertwsmith.ml_pipelines.pipeline.{
  PipelineModelFitConfig,
  ModelFit => PModelFit
}

object ModelFit {

  def main(args: Array[String]): Unit = {
    // Step #1 - Parse the command line arguments
    val parsed: PipelineModelFitConfig =
      PModelFit.parser.parse(args.toSeq, PipelineModelFitConfig()) match {
        case Some(c) => c
        case None =>
          throw new Exception(s"Malformed command line arguments: ${args}")
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
      Array[PipelineStage](
        stringIndexerModel,
        vectorAssembler,
        xgboostClassifier,
        indexToString
      )
    )
    val pipelineModel = pipeline.fit(trainDF)

    if (parsed.overwrite)
      pipelineModel.write.overwrite().save(parsed.pipeline)
    else
      pipelineModel.save(parsed.pipeline)

    val xgboostClassificationModel =
      pipelineModel.stages(2).asInstanceOf[XGBoostClassificationModel]

    // Step #14 - Evaluate RandomForestClassificationModel metrics
    println(
      XGBoostMetricsReport(
        xgboostClassificationModel,
        vectorAssembler.getInputCols
      ).toString
    )

    // Step #15 - Stop Spark
    spark.stop()
  }

}
