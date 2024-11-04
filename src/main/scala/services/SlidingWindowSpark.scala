package services

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.api.ndarray.INDArray
import org.apache.spark.sql.types._

import java.nio.file.{Files, Paths}
import java.net.URI
import com.typesafe.config.ConfigFactory

object SlidingWindowSpark {

  private val config = ConfigFactory.load()

  def main(args: Array[String]): Unit = {

    val embeddingsFilePath = config.getString("filePaths.embeddingsPath")
    val tokensFilePath = config.getString("filePaths.tokensPath")
    val outputPath = config.getString("filePaths.slidingWindowOutputPath")

    val isLocal = if (args.length == 1) args(0).toBoolean else false

    if (!isLocal && args.length < 3) {
      System.err.println("Usage: SlidingWindowSpark <embeddingsPath> <tokensPath> <outputPath>")
      System.exit(1)
    }

    val embeddingsPath = if (isLocal) embeddingsFilePath else args(0)
    val tokensPath = if (isLocal) tokensFilePath else args(1)

    val spark = if (isLocal) {
      SparkSession.builder()
        .appName("SlidingWindowSpark")
        .master("local[*]")
        .config("spark.executor.memory", "2g")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    } else {
      SparkSession.builder()
        .appName("SlidingWindowSpark")
        .config("spark.executor.memory", "4g")
        .config("spark.driver.memory", "2g")
        .config("spark.eventLog.enabled", "true")
        .config("spark.eventLog.dir", "hdfs:///user/akhil/spark-events")
        .getOrCreate()
    }

    val sc = spark.sparkContext
    import spark.implicits._

    def loadEmbeddings(spark: SparkSession, embeddingsPath: String): Map[Int, INDArray] = {
      try {
        val embeddingsSchema = StructType(Seq(
          StructField("tokenId", IntegerType, false)
        ) ++ (1 to 50).map(i => StructField(s"embedding$i", DoubleType, true)))

        val embeddingsDF = spark.read
          .option("header", "true")
          .schema(embeddingsSchema)
          .csv(embeddingsPath)
          .cache()

        val embeddingColumns = embeddingsDF.columns.filter(_.startsWith("embedding"))

        val embeddingsMap = embeddingsDF
          .select("tokenId", embeddingColumns: _*)
          .rdd
          .mapPartitions { partition =>
            partition.map { row =>
              try {
                val tokenId = row.getAs[Int]("tokenId")
                val embeddingsArray = embeddingColumns.map(col =>
                  Option(row.getAs[Double](col)).map(_.toFloat).getOrElse(0.0f)
                ).toArray
                (tokenId, Nd4j.create(embeddingsArray))
              } catch {
                case e: Exception =>
                  println(s"Error processing embedding row: ${e.getMessage}")
                  null
              }
            }.filter(_ != null)
          }
          .collectAsMap()
          .toMap

        embeddingsMap
      } catch {
        case e: Exception =>
          println(s"Error loading embeddings: ${e.getMessage}")
          Map.empty[Int, INDArray]
      }
    }

    def loadTokens(spark: SparkSession, tokensPath: String): Array[Int] = {
      try {
        val tokensSchema = StructType(Seq(
          StructField("tokens", StringType, false)
        ))

        val tokenIds = spark.read
          .option("header", "true")
          .schema(tokensSchema)
          .csv(tokensPath)
          .select("tokens")
          .as[String]
          .flatMap { encodedTokensStr =>
            try {
              encodedTokensStr.trim
                .stripPrefix("[").stripSuffix("]")
                .split("[,\\s]+")
                .filter(_.nonEmpty)
                .map(_.trim.toInt)
            } catch {
              case e: Exception =>
                println(s"Error parsing tokens string: ${e.getMessage}")
                Array.empty[Int]
            }
          }
          .collect()

        tokenIds
      } catch {
        case e: Exception =>
          println(s"Error loading tokens: ${e.getMessage}")
          Array.empty[Int]
      }
    }

    def createInputEmbeddings(
                               inputTokenIds: Array[Int],
                               embeddingsMap: Map[Int, INDArray],
                               embeddingDim: Int
                             ): INDArray = {
      try {
        val sequenceLength = inputTokenIds.length
        val inputEmbeddings = Nd4j.create(sequenceLength, embeddingDim)

        for (i <- inputTokenIds.indices) {
          val tokenEmbedding = getTokenEmbedding(inputTokenIds(i), embeddingsMap, embeddingDim)
          if (tokenEmbedding != null) {
            val positionalEmbedding = computePositionalEmbedding(i, embeddingDim)
            if (positionalEmbedding != null) {
              val tokenEmbeddingCopy = tokenEmbedding.dup()
              val positionalEmbeddingCopy = positionalEmbedding.dup()

              if (tokenEmbeddingCopy.length() == positionalEmbeddingCopy.length()) {
                val combinedEmbedding = tokenEmbeddingCopy.add(positionalEmbeddingCopy)
                if (combinedEmbedding != null) {
                  inputEmbeddings.putRow(i, combinedEmbedding)
                }
              }
            }
          }
        }

        inputEmbeddings
      } catch {
        case e: Exception =>
          println(s"Error in createInputEmbeddings: ${e.getMessage}")
          e.printStackTrace()
          Nd4j.create(1, embeddingDim)
      }
    }

    def getTokenEmbedding(tokenId: Int, embeddingsMap: Map[Int, INDArray], embeddingDim: Int): INDArray = {
      try {
        embeddingsMap.get(tokenId) match {
          case Some(embedding) if embedding != null => embedding.dup()
          case _ => Nd4j.zeros(embeddingDim)
        }
      } catch {
        case e: Exception =>
          println(s"Error getting embedding for token $tokenId: ${e.getMessage}")
          Nd4j.zeros(embeddingDim)
      }
    }

    def computePositionalEmbedding(position: Int, dModel: Int): INDArray = {
      try {
        val embedding = Nd4j.create(dModel)
        for (i <- 0 until dModel) {
          val angle = position.toDouble / math.pow(10000, (2 * (i / 2)) / dModel.toDouble)
          val value = if (i % 2 == 0) math.sin(angle) else math.cos(angle)
          embedding.putScalar(i, value.toFloat)
        }
        embedding
      } catch {
        case e: Exception =>
          println(s"Error computing positional embedding for position $position: ${e.getMessage}")
          Nd4j.zeros(dModel)
      }
    }

    def createTrainingData(row: org.apache.spark.sql.Row, embeddingsMap: Map[Int, INDArray], embeddingDim: Int): (Array[Byte], Array[Byte]) = {
      try {
        val inputTokenIds = row.getAs[Seq[Int]]("inputTokens").toArray
        val targetTokenId = row.getAs[Int]("targetTokenId")

        val inputEmbeddings = createInputEmbeddings(inputTokenIds, embeddingsMap, embeddingDim)
        val targetEmbedding = getTokenEmbedding(targetTokenId, embeddingsMap, embeddingDim)

        if (inputEmbeddings != null && targetEmbedding != null) {
          (Nd4j.toByteArray(inputEmbeddings), Nd4j.toByteArray(targetEmbedding))
        } else {
          throw new Exception("Null embeddings encountered")
        }
      } catch {
        case e: Exception =>
          println(s"Error creating training data: ${e.getMessage}")
          (new Array[Byte](0), new Array[Byte](0))
      }
    }

    try {
      println("Loading embeddings...")
      val embeddingsMap = loadEmbeddings(spark, embeddingsPath)
      if (embeddingsMap.isEmpty) {
        throw new Exception("Failed to load embeddings")
      }
      val embeddingsMapBroadcast = sc.broadcast(embeddingsMap)
      println(s"Loaded ${embeddingsMap.size} embeddings")

      println("Loading tokens...")
      val tokenIds = loadTokens(spark, tokensPath)
      if (tokenIds.isEmpty) {
        throw new Exception("Failed to load tokens")
      }
      println(s"Loaded ${tokenIds.length} tokens")

      val windowSize = 4
      val embeddingDim = 50

      println("Creating sliding windows...")
      val tokensDF = tokenIds.toSeq.toDF("tokenId")
        .withColumn("index", monotonically_increasing_id())
        .cache()

      val w = Window.orderBy("index").rowsBetween(-windowSize, -1)

      val slidingDF = tokensDF
        .withColumn("inputTokens", collect_list("tokenId").over(w))
        .withColumn("targetTokenId", col("tokenId"))
        .filter(size(col("inputTokens")) === windowSize)
        .cache()

      val trainingDataRDD = slidingDF.rdd.mapPartitions { partition =>
        partition.map { row =>
          createTrainingData(row, embeddingsMapBroadcast.value, embeddingDim)
        }.filter { case (inputBytes, targetBytes) =>
          inputBytes.length > 0 && targetBytes.length > 0
        }
      }.cache()

      // Create output directories if running locally
      val outputBasePath = if (isLocal) {
        val baseDir = Paths.get(outputPath)
        val objectDir = baseDir.resolve("object")
        val csvDir = baseDir.resolve("csv")

        Files.createDirectories(baseDir)
        Files.createDirectories(objectDir)
        Files.createDirectories(csvDir)

        outputPath
      } else {
        s"hdfs:///$outputPath/"
      }

      // Construct proper paths for local or HDFS
      val objectFilePath = if (isLocal) {
        new org.apache.hadoop.fs.Path(s"file://${Paths.get(outputBasePath, "object").toAbsolutePath}")
      } else {
        new org.apache.hadoop.fs.Path(outputBasePath + "object")
      }

      val csvPath = if (isLocal) {
        new org.apache.hadoop.fs.Path(s"file://${Paths.get(outputBasePath, "csv").toAbsolutePath}")
      } else {
        new org.apache.hadoop.fs.Path(outputBasePath + "csv")
      }

      // Configure filesystem
      val fs = if (isLocal) {
        sc.hadoopConfiguration.set("fs.file.impl", classOf[org.apache.hadoop.fs.LocalFileSystem].getName)
        sc.hadoopConfiguration.set("fs.file.impl.disable.cache", "true")
        org.apache.hadoop.fs.FileSystem.get(new URI("file:///"), sc.hadoopConfiguration)
      } else {
        org.apache.hadoop.fs.FileSystem.get(new URI("hdfs:///"), sc.hadoopConfiguration)
      }

      // Clean up existing paths
      Seq(objectFilePath, csvPath).foreach { path =>
        if (fs.exists(path)) {
          println(s"Deleting existing path: $path")
          fs.delete(path, true)
        }
      }

      // Save outputs
      println("Saving training data...")
      println(s"Saving to object file path: ${objectFilePath.toString}")
      println(s"Saving to CSV path: ${csvPath.toString}")

      trainingDataRDD.coalesce(4).saveAsObjectFile(objectFilePath.toString)
      trainingDataRDD.coalesce(1).saveAsTextFile(csvPath.toString)

      println(s"Training data saved successfully to: $outputBasePath")

    } catch {
      case e: Exception =>
        println(s"Error during processing: ${e.getMessage}")
        e.printStackTrace()
        throw e
    } finally {
      if (sc != null) {
        try {
          sc.getPersistentRDDs.foreach { case (_, rdd) =>
            try {
              rdd.unpersist()
            } catch {
              case _: Exception => // Ignore unpersist errors
            }
          }
        } catch {
          case _: Exception => // Ignore cleanup errors
        }
      }
      spark.stop()
    }
  }
}