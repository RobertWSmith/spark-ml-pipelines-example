package com.github.robertwsmith.ml_pipelines.no_pipeline

import com.github.robertwsmith.ml_pipelines._
import org.apache.spark.ml.PipelineStage
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler}
import org.apache.spark.ml.util.MLWritable
import org.apache.spark.sql.SparkSession
import scopt.OptionParser

case class NoPipelineModelFitConfig(
  input: String = "examples/data/iris-train",
  stringIndexer: String = "examples/model/no_pipeline/string_indexer",
  vectorAssembler: String = "examples/model/no_pipeline/vector_assembler",
  randomForest: String = "examples/model/no_pipeline/random_forest",
  indexToString: String = "examples/model/no_pipeline/index_to_string",
  overwrite: Boolean = false
)

object ModelFit {

  val name: String = "no-pipeline-model-fit"

  val parser: OptionParser[NoPipelineModelFitConfig] =
    new OptionParser[NoPipelineModelFitConfig](name) {
      head(name, "1.0")

      opt[String]("input")
        .action((x, c) => c.copy(input = x))
        .optional()

      opt[String]("string_indexer")
        .action((x, c) => c.copy(stringIndexer = x))
        .optional()

      opt[String]("vector_assembler")
        .action((x, c) => c.copy(vectorAssembler = x))
        .optional()

      opt[String]("random_forest")
        .action((x, c) => c.copy(randomForest = x))
        .optional()

      opt[String]("index_to_string")
        .action((x, c) => c.copy(indexToString = x))
        .optional()

      opt[Unit]("overwrite")
        .action((_, c) => c.copy(overwrite = true))
        .optional()
    }

  /** Machine Learning Pipeline Tutorial - No Pipeline Example
    *
    * 1. Parse the command line arguments
    * 2. Start SparkSession
    * 3. Read in the training dataset
    * 4. Create StringIndexer instance, then fit a StringIndexerModel
    * 5. Write StringIndexerModel to HDFS
    * 6. Create VectorAssembler instance
    * 7. Write VectorAssembler to HDFS
    * 8. Create RandomForestClassifier instance
    * 9. Create IndexToString instance
    * 10. Write IndexToString to HDFS
    * 11. Apply StringIndexerModel, VectorAssembler transformations to the training dataset
    * 12. Fit the RandomForestClassifier estimator (generate a RandomForestClassificationModel)
    * 13. Write RandomForestClassificationModel to HDFS
    * 14. Evaluate RandomForestClassificationModel metrics
    * 15. Stop SparkSession
    */
  def main(args: Array[String]): Unit = {
    // Step #1 - Parse the command line arguments
    val parsed: NoPipelineModelFitConfig =
      parser.parse(args, NoPipelineModelFitConfig()) match {
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

    // Step #5 - Write StringIndexerModel to HDFS
    this.persistPipelineStage(stringIndexerModel, parsed.stringIndexer, parsed.overwrite)

    // Step #6 - Create VectorAssembler Instance
    val vectorAssembler = new VectorAssembler()
      .setInputCols(predictorColumnNames)
      .setOutputCol(featureColumnName)

    // Step #7 - Write VectorAssember to HDFS
    persistPipelineStage(vectorAssembler, parsed.vectorAssembler, parsed.overwrite)

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

    // Step #10 - Write IndexToString to HDFS
    persistPipelineStage(indexToString, parsed.indexToString, parsed.overwrite)

    // Step #11 - Apply StringIndexerModel, VectorAssembler transformations to the training dataset
    val stringTransformed = stringIndexerModel.transform(trainDF)
    val vectorAssembled   = vectorAssembler.transform(stringTransformed)

    // Step #12 - Fit the RandomForestClassifier estimator
    val randomForestClassificationModel = randomForest.fit(vectorAssembled)

    // Step #13 - Write RandomForestClassificationModel to HDFS
    persistPipelineStage(randomForestClassificationModel, parsed.randomForest, parsed.overwrite)

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

  def persistPipelineStage(
    stage: PipelineStage with MLWritable,
    path: String,
    overwrite: Boolean = false
  ): Unit = {
    if (overwrite)
      stage.write.overwrite().save(path)
    else
      stage.save(path)
  }

}
