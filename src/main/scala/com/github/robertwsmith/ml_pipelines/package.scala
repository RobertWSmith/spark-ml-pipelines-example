package com.github.robertwsmith

import ml.dmlc.xgboost4j.scala.spark.XGBoostClassificationModel
import org.apache.spark.ml.classification.RandomForestClassificationModel
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.types.{
  DoubleType,
  StringType,
  StructField,
  StructType
}
import org.apache.spark.sql.{DataFrame, SaveMode}

import scala.collection.immutable.StringOps
import scala.collection.mutable.ArrayBuffer

package object ml_pipelines {

  val targetColumnName: String = "species"
  val labelColumnName: String = "label"
  val featureColumnName: String = "features"
  val predictorColumnNames: Array[String] =
    Array("sepal_length", "sepal_width", "petal_length", "petal_width")

  val multiclassMetrics: Array[String] =
    Array("accuracy", "f1", "weightedPrecision", "weightedRecall")
  lazy val maxMetricLength: Int = multiclassMetrics.map(_.length).max

  val irisSchema: StructType = StructType(
    StructField("sepal_length", DoubleType) ::
      StructField("sepal_width", DoubleType) ::
      StructField("petal_length", DoubleType) ::
      StructField("petal_width", DoubleType) ::
      StructField("species", StringType) ::
      Nil
  )

  def makeSaveMode(overwrite: Boolean): SaveMode = {
    if (overwrite)
      SaveMode.Overwrite
    else
      SaveMode.ErrorIfExists
  }

  case class EvaluateMetric(
      metric: String,
      dataFrame: DataFrame,
      labelCol: String,
      predictionCol: String
  ) {
    require(multiclassMetrics.contains(metric))

    def calculate(): Double =
      new MulticlassClassificationEvaluator()
        .setLabelCol(labelCol)
        .setPredictionCol(predictionCol)
        .setMetricName(metric)
        .evaluate(dataFrame)
  }

  case class ModelFitMetrics(
      dataFrame: DataFrame,
      labelCol: String,
      predictionCol: String
  ) {
    def calculate(): Map[String, Double] = {
      val buffer = new ArrayBuffer[(String, Double)]()

      multiclassMetrics.foreach(mcm => {
        buffer.append(
          (
            mcm,
            EvaluateMetric(mcm, dataFrame, labelCol, predictionCol).calculate()
          )
        )
      })

      buffer.toMap
    }

    override def toString: String = {
      val builder = new StringBuilder

      builder.append("Model Fit Metrics: \n")
      val calcs = calculate()
      val maxLen = calcs.keys.map(_.length).max
      calcs.foreach {
        case (key: String, value: Double) => {
          val padding =
            if (maxLen == key.length) ""
            else new StringOps(" ") * (maxLen - key.length)
          builder.append(s"\t${key}${padding} -> ${value}\n")
        }
      }
      builder.toString()
    }
  }

  case class RandomForestMetricsReport(
      model: RandomForestClassificationModel,
      inputCols: Array[String]
  ) {
    lazy val maxInputColLength: Int = inputCols.map(_.length).max

    override def toString: String = {
      val builder = new StringBuilder()

      builder.append("Random Forest Model Metrics: \n")
      builder.append(s"Number of Trees:       ${model.getNumTrees}\n")
      builder.append(s"Number of Classes:     ${model.getNumTrees}\n")
      builder.append(s"Number of Features:    ${model.getNumTrees}\n")
      builder.append(s"Total Number of Nodes: ${model.getNumTrees}\n")

      val features = model.featureImportances.toArray.zipWithIndex.toSeq
        .sortBy(_._1)(Ordering[Double].reverse)
      val maxLen = features.map(x => inputCols(x._2).length).max

      features.foreach { case (fi, idx) =>
        val name = inputCols(idx)
        val padding =
          if (maxLen == name.length) ""
          else new StringOps(" ") * (maxLen - name.length)
        builder.append(s"\t${name}${padding} -> ${fi}\n")
      }

      builder.toString()
    }
  }

  case class XGBoostMetricsReport(
      model: XGBoostClassificationModel,
      inputCols: Array[String],
      importanceType: String = "gain"
  ) {

    override def toString: String = {
      val builder = new StringBuilder()

      builder.append("XGBoost Classification Model Metrics: \n")

      builder.append(s"Objective:         ${model.getObjective}\n")
      builder.append(s"ETA:               ${model.getObjective}\n")
      builder.append(s"Alpha:             ${model.getObjective}\n")
      builder.append(s"Number of Classes: ${model.getObjective}\n")
      builder.append(s"Maximum Bins:      ${model.getObjective}\n")
      builder.append(s"Maximum Depth:     ${model.getObjective}\n")

      val maxInputColLength = inputCols.map(_.length).max

      val featureScoreMap =
        model.nativeBooster.getScore(inputCols, importanceType)

      builder.append("Feature Importances: \n")

      featureScoreMap.toSeq
        .sortBy(_._2)(Ordering[Double].reverse)
        .foreach {
          case (key: String, value: Double) => {
            val padding =
              if (maxInputColLength == key.length) ""
              else new StringOps(" ") * (maxInputColLength - key.length)
            builder.append(s"\t${key}${padding} -> ${value}\n")
          }
        }

      builder.toString()
    }
  }

}
