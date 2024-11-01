import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._
import utils.SlidingWindowWithEmbeddings
import com.typesafe.config.ConfigFactory
import org.nd4j.linalg.factory.Nd4j
import java.nio.file.{Files, Paths}
import org.apache.hadoop.fs.{FileSystem, Path}


object SlidingWindowSpark {

//  private val config = ConfigFactory.load()


  def main(args: Array[String]): Unit = {
    // Set up Spark session
    val spark = SparkSession.builder()
      .appName("SlidingWindowSpark")
      .master("local[*]")
      .getOrCreate()

    val sc = spark.sparkContext
    import spark.implicits._

//    val embeddingsFilePath = config.getString("filePaths.embeddingsPath")

    // Load embeddings and tokens as before
    val embeddingsMap =  SlidingWindowWithEmbeddings.loadEmbeddings("src/main/resources/Files/embeddings.csv")
    val embeddingsMapBroadcast = sc.broadcast(embeddingsMap)

//    val tokensFilePath = config.getString("filePaths.tokensPath")

    val tokenIds = SlidingWindowWithEmbeddings.loadTokens("src/main/resources/Files/output.csv")

    // Parameters
    val windowSize = 4
    val embeddingDim = 50 // Ensure this matches your embeddings' dimensions

    // Create DataFrame of tokens with index
    val tokensDF = tokenIds.toSeq.toDF("tokenId").withColumn("index", monotonically_increasing_id())

    // Define window specification
    val w = Window.orderBy("index").rowsBetween(-windowSize, -1)

    // Create sliding windows
    val slidingDF = tokensDF
      .withColumn("inputTokens", collect_list("tokenId").over(w))
      .withColumn("targetTokenId", col("tokenId"))
      .filter(size(col("inputTokens")) === windowSize)

    // Map to embeddings
    val trainingDataRDD = slidingDF.rdd.map { row =>
      val inputTokenIds = row.getAs[Seq[Int]]("inputTokens").toArray
      val targetTokenId = row.getAs[Int]("targetTokenId")

      val embeddingsMap = embeddingsMapBroadcast.value
      val inputEmbeddings = SlidingWindowWithEmbeddings.createInputEmbeddings(inputTokenIds, embeddingsMap, embeddingDim)
      val targetEmbedding = SlidingWindowWithEmbeddings.getTokenEmbedding(targetTokenId, embeddingsMap, embeddingDim)
      (inputEmbeddings, targetEmbedding)
    }

    // Serialize INDArray objects to byte arrays
    val serializableRDD = trainingDataRDD.map { case (inputEmbeddings, targetEmbedding) =>
      val inputBytes = Nd4j.toByteArray(inputEmbeddings)
      val targetBytes = Nd4j.toByteArray(targetEmbedding)
      (inputBytes, targetBytes)
    }

    val path = "src/main/resources/trainingData/"
    val csvPath = "src/main/resources/trainingDataCSV/"
    val fs = FileSystem.get(sc.hadoopConfiguration)
    val outputPath = new Path(path)
    if (fs.exists(outputPath)) {
      println(s"Path $path exists before saving.")
    } else {
      println(s"Path $path does not exist before saving.")
    }

    // 2. Clear the path if it exists
    if (fs.exists(outputPath)) {
      fs.delete(outputPath, true)
      println(s"Deleted existing path: $path")
    }

    // Save the RDD as an object file
    serializableRDD.saveAsObjectFile(path)
    serializableRDD.coalesce(1).saveAsTextFile(csvPath)


    // Stop Spark session
    spark.stop()
  }

  // Existing functions: loadEmbeddings, loadTokens, createInputEmbeddings, getTokenEmbedding, computePositionalEmbedding
}
