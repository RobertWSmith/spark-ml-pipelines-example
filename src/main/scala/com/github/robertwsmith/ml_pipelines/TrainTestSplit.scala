package com.github.robertwsmith.ml_pipelines

import org.apache.spark.sql.{DataFrame, SparkSession}
import scopt.OptionParser

case class TrainTestSplitConfig(
  input: String = "examples/data/iris-raw",
  train: String = "examples/data/iris-train",
  test: String = "examples/data/iris-test",
  trainPercent: Double = 0.7,
  overwrite: Boolean = false
)

object TrainTestSplit {
  val name: String = "train-test-split"

  val parser: OptionParser[TrainTestSplitConfig] =
    new OptionParser[TrainTestSplitConfig](name) {
      head(name, "1.0")

      opt[String]("input")
        .action((x, c) => c.copy(input = x))
        .optional()

      opt[String]("train")
        .action((x, c) => c.copy(train = x))
        .optional()

      opt[String]("test")
        .action((x, c) => c.copy(test = x))
        .optional()

      opt[Double]("train_percent")
        .action((x, c) => c.copy(trainPercent = x))
        .optional()

      opt[Unit]("overwrite")
        .action((_, c) => c.copy(overwrite = true))
        .optional()
    }

  /** Perform the train/test split and persist to disk
    *
    * Steps:
    * 1. Parse the command line arguments
    * 2. Start SparkSession
    * 3. Read in the DataFrame
    * 4. Split into two DataFrames based on train/test split percents
    * 5. Write Train & Test to their respective directories as Parquet files
    * 6. Stop SparkSession
    *
    * @param args command line arguments
    */
  def main(args: Array[String]): Unit = {
    // Step #1 - Parse the command line arguments
    val parsed: TrainTestSplitConfig =
      parser.parse(args, TrainTestSplitConfig()) match {
        case Some(c) => require(c.train != c.test); c
        case None =>
          throw new Exception(s"Malformed command line arguments: ${args.mkString(", ")}")
      }

    // Step #2 - Start SparkSession
    val spark: SparkSession = SparkSession.builder.getOrCreate()

    // Step #3 - Read in the source DataFrame
    val irisDF: DataFrame = spark.read.schema(irisSchema).csv(parsed.input)
  }
}
