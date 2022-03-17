package com.github.robertwsmith

import ml.dmlc.xgboost4j.scala.spark.XGBoostClassificationModel
import org.apache.spark.ml.classification.RandomForestClassificationModel
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.{DataFrame, SaveMode}
import org.apache.spark.sql.types.{
  DoubleType,
  StringType,
  StructField,
  StructType
}

import scala.collection.immutable.StringOps
import scala.collection.mutable.ArrayBuffer

package object ml_pipelines {

  val targetColumnName: String = "species"
  val labelColumnName: String = "label"
  val featureColumnName: String = "features"
  val predictorColumnName: Array[String] =
    Array("sepal_length", "sepal_width", "petal_length", "petal_width")

  val multiclassMetrics: Array[String] =
    Array("accuracy", "f1", "weightedPrecision", "weightedRecall")
  val maxMetricLen: Int = multiclassMetrics.map(_.length).max

  case class RandomForestClassificationModelMetrics(
      model: RandomForestClassificationModel,
      inputCols: Array[String]
  ) {
    lazy val maxInputColLength: Int = inputCols.map(_.length).max

    override def toString: String = {
      val builder = new StringBuilder()
      builder.append("Random Forest Metrics: \n")
      builder.append(s"Number of Trees:       ${model.getNumTrees}\n")
      builder.append(s"Number of Classes:     ${model.numClasses}\n")
      builder.append(s"Number of Features:    ${model.numFeatures}\n")
      builder.append(s"Total Number of Nodes: ${model.totalNumNodes}\n")

      this.buildFeatureImportances(builder)
      builder.toString()
    }

    /** Utilizing a passed reference example. Modifying the 'builder'
      * will result in the calling environment's 'builder' being modified.
      *
      * @param builder [[StringBuilder]] that collects feature importances
      */
    def buildFeatureImportances(builder: StringBuilder): Unit = {
      builder.append("Feature Importances: \n")
      this.featureImportances().foreach {
        case (key: String, value: Double) => {
          val padding: String =
            if (maxInputColLength == key.length) ""
            else new StringOps(" ") * (maxInputColLength - key.length)

          builder.append(s"\t${key}${padding} -> ${value}\n")
        }
      }
    }

    /** Convert the raw feature importances into a pre-sorted sequence
      * ordered from most to least important.
      *
      * @return [[ Seq[(String, Double)] ]] of feature column names and importance values
      */
    def featureImportances(): Seq[(String, Double)] = {
      val buffer = new ArrayBuffer[(String, Double)]()
      model.featureImportances.toArray.zipWithIndex.toSeq
        .sortBy(_._1)(Ordering[Double].reverse)
        .foreach { case (fi, idx) =>
          buffer.append((inputCols(idx), fi))
        }
      buffer
    }

  }

  val irisSchema: StructType = StructType(
    StructField("sepal_length", DoubleType) ::
      StructField("sepal_width", DoubleType) ::
      StructField("petal_length", DoubleType) ::
      StructField("petal_width", DoubleType) ::
      StructField("species", StringType) ::
      Nil
  )

  def evaluateMetrics(
      metric: String,
      dataFrame: DataFrame,
      labelCol: String,
      predictionCol: String
  ): Double = {
    require(multiclassMetrics.contains(metric))

    new MulticlassClassificationEvaluator()
      .setLabelCol(labelCol)
      .setPredictionCol(predictionCol)
      .setMetricName(metric)
      .evaluate(dataFrame)
  }

  def modelFitMetrics(
      dataFrame: DataFrame,
      labelCol: String,
      predictionCol: String
  ): Map[String, Double] = {
    val buffer = new ArrayBuffer[(String, Double)]()

    multiclassMetrics.foreach(mcm => {
      buffer.append(
        (mcm, evaluateMetrics(mcm, dataFrame, labelCol, predictionCol))
      )
    })

    buffer.toMap
  }

  def makeSaveMode(overwrite: Boolean): SaveMode = {
    if (overwrite)
      SaveMode.Overwrite
    else
      SaveMode.ErrorIfExists
  }

  case class XGBoostClassificationModelMetrics(
      model: XGBoostClassificationModel,
      inputCols: Array[String],
      importanceType: String = "gain"
  ) {
    lazy val maxInputColLen: Int = inputCols.map(_.length).max

    override def toString: String = {
      val builder = new StringBuilder()

      builder.append("XGBoost Classification Model Metrics: \n")

      builder.append(s"Objective: ${model.getObjective}")
      builder.append(s"ETA: ${model.getEta}")
      builder.append(s"Alpha: ${model.getAlpha}")
      builder.append(s"Number of Classes: ${model.getNumClass}")
      builder.append(s"Maximum Bins: ${model.getMaxBins}")
      builder.append(s"Maximum Depth: ${model.getMaxDepth}")

      this.featureImportances(builder)

      builder.toString()
    }

    private def featureImportances(builder: StringBuilder): Unit = {
      val featureScoreMap: Map[String, Double] =
        model.nativeBooster.getScore(inputCols, importanceType)

      builder.append("\nFeature Importances: \n")
      (featureScoreMap.toList
        .sortBy(_._2)(Ordering[Double].reverse))
        .foreach {
          case (key, value) => {
            val padding: String =
              if (maxInputColLen == key.length) ""
              else new StringOps(" ") * (maxInputColLen - key.length)
            builder.append(s"\t${key}${padding} -> ${value}\n")
          }
        }
    }
  }

}
