package com.github.robertwsmith.ml_pipelines.no_pipeline

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
import org.apache.spark.sql.SparkSession
import scopt.OptionParser

case class NoPipelinePredictConfig(
    input: String = "examples/data/iris-test",
    output: String = "examples/data/no_pipeline/iris-test-predict",
    stringIndexer: String = "examples/model/no_pipeline/string_indexer",
    vectorAssembler: String = "examples/model/no_pipeline/vector_assembler",
    randomForest: String = "examples/model/no_pipeline/random_forest",
    indexToString: String = "examples/model/no_pipeline/index_to_string",
    overwrite: Boolean = false
)

class Predict {
  val name: String = "no-pipeline-predict"

  val parser = new OptionParser[NoPipelinePredictConfig](name) {
    head(name, "1.0")

    opt[String]("input")
      .action((x, c) => c.copy(input = x))
      .optional()

    opt[String]("output")
      .action((x, c) => c.copy(output = x))
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

  /** No Pipeline Model Predict
    *
    * 1. Parse the command line arguments.
    * 2. Start SparkSession
    * 3. Load StringIndexerModel, VectorAssembler, RandomForestClassificationModel & IndexToString
    * 4. Read in the test DataFrame
    * 5. Use the models loaded to transform the DataFrame and apply the fitted model prediction
    * 6. Save the prediction DataFrame
    * 7. Run the model fit metrics
    * 8. Stop SparkSession
    *
    * @param args command line arguments
    */
  def main(args: Array[String]): Unit = {
    // Step 1
    val parsed: NoPipelinePredictConfig =
      this.parser.parse(args, NoPipelinePredictConfig()) match {
        case Some(c) => c
        case None =>
          throw new Exception(s"Malformed command line arguments: ${args}")
      }

    // Step 2
    val spark = SparkSession.builder().getOrCreate()

    // Step 3
    val stringIndexer = StringIndexerModel.load(parsed.stringIndexer)
    val vectorAssembler = VectorAssembler.load(parsed.vectorAssembler)
    val randomForest = RandomForestClassificationModel.load(parsed.randomForest)
    val indexToString = IndexToString.load(parsed.indexToString)

    // Step 4
    val testDF = spark.read.parquet(parsed.input)

    // Step 5
    val testStringIndexed = stringIndexer.transform(testDF)
    val testVectorAssembled = vectorAssembler.transform(testStringIndexed)
    val testRandomForest = randomForest.transform(testVectorAssembled)
    val prediction = indexToString.transform(testRandomForest)

    // Step 6
    prediction.write.mode(makeSaveMode(parsed.overwrite)).parquet(parsed.output)

    // Step 7
    val labelCol = randomForest.getOrDefault(randomForest.labelCol)
    val predictionCol = randomForest.getOrDefault(randomForest.predictionCol)
    println(ModelFitMetrics(prediction, labelCol, predictionCol).toString)

    // Step 8
    spark.stop()
  }

}
