package com.github.robertwsmith.ml_pipelines.pipeline

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

case class PipelineModelFitConfig(
    input: String = "examples/data/iris-train",
    pipeline: String = "examples/model/pipeline/",
    overwrite: Boolean = false
)

object ModelFit {

  val name: String = "pipeline-model-fit"

  val parser: OptionParser[PipelineModelFitConfig] =
    new OptionParser[PipelineModelFitConfig](name) {
      head(name, "1.0")

      opt[String]("input")
        .action((x, c) => c.copy(input = x))
        .optional()

      opt[String]("pipeline")
        .action((x, c) => c.copy(pipeline = x))
        .optional()

      opt[Unit]("overwrite")
        .action((_, c) => c.copy(overwrite = true))
        .optional()
    }

  /** Machine Learning Pipeline Tutorial - Pipeline Example
    */
  def main(args: Array[String]): Unit = {
    // Step #1 - Parse the command line arguments
    val parsed: PipelineModelFitConfig =
      parser.parse(args, PipelineModelFitConfig()) match {
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
      Array[PipelineStage](
        stringIndexerModel,
        vectorAssembler,
        randomForest,
        indexToString
      )
    )
    val pipelineModel = pipeline.fit(trainDF)

    if (parsed.overwrite)
      pipelineModel.write.overwrite().save(parsed.pipeline)
    else
      pipelineModel.save(parsed.pipeline)

    val randomForestClassificationModel =
      pipelineModel.stages(2).asInstanceOf[RandomForestClassificationModel]

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
