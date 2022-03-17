import Dependencies._

organization := "com.github.robertwsmith"

name := "spark-ml-pipelines-example"

version := "1.0.0"

scalaVersion := "2.11.12"

libraryDependencies ++= Seq(
  scopt,
  spark_core,
  spark_hive,
  spark_mllib,
  spark_sql,
  xgboost4j_spark,
  hadoop_common,
  hive_common,
  json4s_core,
  slf4j,
  scalatest
)