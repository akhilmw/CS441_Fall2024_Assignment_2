package services

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers
import org.nd4j.linalg.factory.Nd4j
import org.apache.spark.sql.Row

class SlidingWindowSparkTest extends AnyFunSuite with Matchers {

  // Initialize SparkSession for testing
  val spark = SparkSession.builder()
    .appName("SlidingWindowSparkTest")
    .master("local[*]")
    .getOrCreate()

  import spark.implicits._

  // Dummy paths and configuration for tests
  val embeddingsPath = "src/test/resources/test_embeddings.csv"
  val tokensPath = "src/test/resources/test_tokens.csv"

  test("loadEmbeddings should load embeddings correctly") {
    // Create a DataFrame for testing
    val data = Seq(
      Row(1, 0.1, 0.2, 0.3),
      Row(2, 0.4, 0.5, 0.6)
    )
    val schema = StructType(Seq(
      StructField("tokenId", IntegerType, nullable = false),
      StructField("embedding1", DoubleType, nullable = true),
      StructField("embedding2", DoubleType, nullable = true),
      StructField("embedding3", DoubleType, nullable = true)
    ))

    val embeddingsDF = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    embeddingsDF.write.option("header", "true").csv(embeddingsPath)

    // Call the method and check the result
    val result = SlidingWindowSpark.loadEmbeddings(spark, embeddingsPath)
    result should have size 2
    result(1) shouldEqual Nd4j.create(Array(0.1f, 0.2f, 0.3f))
    result(2) shouldEqual Nd4j.create(Array(0.4f, 0.5f, 0.6f))
  }

  test("loadTokens should load tokens correctly") {
    // Create a DataFrame for testing
    val data = Seq(
      Row("[1, 2, 3, 4]"),
      Row("[5, 6, 7, 8]")
    )
    val schema = StructType(Seq(
      StructField("tokens", StringType, nullable = false)
    ))

    val tokensDF = spark.createDataFrame(spark.sparkContext.parallelize(data), schema)
    tokensDF.write.option("header", "true").csv(tokensPath)

    // Call the method and check the result
    val result = SlidingWindowSpark.loadTokens(spark, tokensPath)
    result shouldEqual Array(1, 2, 3, 4, 5, 6, 7, 8)
  }

  test("createInputEmbeddings should create the correct input embeddings") {
    val embeddingsMap = Map(
      1 -> Nd4j.create(Array(0.1f, 0.2f)),
      2 -> Nd4j.create(Array(0.3f, 0.4f))
    )
    val inputTokenIds = Array(1, 2)
    val embeddingDim = 2

    val result = SlidingWindowSpark.createInputEmbeddings(inputTokenIds, embeddingsMap, embeddingDim)
    result.getRow(0) shouldEqual embeddingsMap(1)
    result.getRow(1) shouldEqual embeddingsMap(2)
  }

  test("computePositionalEmbedding should compute correct values") {
    val dModel = 6
    val result = SlidingWindowSpark.computePositionalEmbedding(1, dModel)

    // Check specific values
    result.getFloat(0) should be(math.sin(1.0 / math.pow(10000, 0.0)).toFloat)
    result.getFloat(1) should be(math.cos(1.0 / math.pow(10000, 0.0)).toFloat)
  }

  test("createTrainingData should return correct embeddings") {
    val embeddingsMap = Map(
      1 -> Nd4j.create(Array(0.1f, 0.2f)),
      2 -> Nd4j.create(Array(0.3f, 0.4f))
    )

    val data = Seq(
      Row(Seq(1, 2), 2)
    )
    val schema = StructType(Seq(
      StructField("inputTokens", ArrayType(IntegerType), nullable = false),
      StructField("targetTokenId", IntegerType, nullable = false)
    ))

    val row = spark.createDataFrame(spark.sparkContext.parallelize(data), schema).collect()(0)

    val (inputBytes, targetBytes) = SlidingWindowSpark.createTrainingData(row, embeddingsMap, 2)

    // Decode the bytes back to verify
    val inputEmbedding = Nd4j.fromByteArray(inputBytes)
    val targetEmbedding = Nd4j.fromByteArray(targetBytes)

    inputEmbedding.getRow(0) shouldEqual embeddingsMap(1)
    inputEmbedding.getRow(1) shouldEqual embeddingsMap(2)
    targetEmbedding shouldEqual embeddingsMap(2)
  }

  // Clean up after tests
  after {
    spark.stop()
  }
}
