ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.12.18"

lazy val root = (project in file("."))
  .settings(
    name := "CS441_Fall2024_Assignment_2"
  )

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "3.4.3",
  "org.apache.spark" %% "spark-sql"  % "3.4.3",
  "org.apache.spark" %% "spark-mllib" % "3.4.3",
  "org.deeplearning4j" % "dl4j-spark_2.12" % "1.0.0-M2.1",
  "org.deeplearning4j" % "deeplearning4j-nlp" % "1.0.0-M2.1",
  "org.deeplearning4j" % "deeplearning4j-core" % "1.0.0-M2.1",
  "org.nd4j" % "nd4j-native-platform" % "1.0.0-M2.1",
  "org.nd4j" % "nd4j-native" % "1.0.0-M2.1",
  "org.slf4j" % "slf4j-simple" % "2.0.13",
  "com.typesafe" % "config" % "1.4.3",
  "org.scalatest" %% "scalatest" % "3.2.19" % "test",
)

