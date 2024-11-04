import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.apache.spark.sql.SparkSession
import org.nd4j.linalg.factory.Nd4j
import org.apache.spark.sql.Row
import org.apache.spark.sql.types._

import java.nio.file.{Files, Path, Paths}
import java.io.File
import org.scalatest.BeforeAndAfterAll
import services.SlidingWindowSpark

class SlidingWindowSparkTest extends AnyFlatSpec with Matchers with BeforeAndAfterAll {

  // Initialize Spark
  val spark: SparkSession = SparkSession.builder()
    .appName("SlidingWindowSparkTest")
    .master("local[*]")
    .getOrCreate()

  import spark.implicits._

  // Create test data
  val testEmbeddingsData: String =
    """tokenId,embedding1,embedding2,embedding3
      |1,0.1,0.2,0.3
      |2,0.4,0.5,0.6
      |3,0.7,0.8,0.9
      |4,0.11,0.12,0.13""".stripMargin

  val testTokensData: String =
    """tokens
      |[1, 2, 3, 4, 1, 2]""".stripMargin

  // Helper method to create temporary files
  def createTempFile(content: String): String = {
    val temp = File.createTempFile("test", ".csv")
    temp.deleteOnExit()
    Files.write(temp.toPath, content.getBytes)
    temp.getAbsolutePath
  }

  "loadEmbeddings" should "correctly load and parse embeddings from CSV" in {
    val embeddingsPath = createTempFile(testEmbeddingsData)
    val embeddingsMap = SlidingWindowSpark.loadEmbeddings(spark, embeddingsPath)

    // Verify the map contains the correct number of entries
    embeddingsMap.size shouldBe 4

    // Verify the content of embeddings
    val firstEmbedding = embeddingsMap(1)
    firstEmbedding.getFloat(0L) shouldBe 0.1f +- 0.001f
    firstEmbedding.getFloat(1L) shouldBe 0.2f +- 0.001f
    firstEmbedding.getFloat(2L) shouldBe 0.3f +- 0.001f
  }

  "computePositionalEmbedding" should "generate correct positional embeddings" in {
    val position = 0
    val dModel = 4
    val embedding = SlidingWindowSpark.computePositionalEmbedding(position, dModel)

    // Verify embedding dimensions
    embedding.length() shouldBe dModel

    // Verify the first value (sin for position 0)
    embedding.getFloat(0L) shouldBe 0.0f +- 0.001f
  }

  "getTokenEmbedding" should "return correct embedding for known token" in {
    val embeddingsPath = createTempFile(testEmbeddingsData)
    val embeddingsMap = SlidingWindowSpark.loadEmbeddings(spark, embeddingsPath)
    val embedding = SlidingWindowSpark.getTokenEmbedding(1, embeddingsMap, 3)

    // Verify embedding values
    embedding.getFloat(0L) shouldBe 0.1f +- 0.001f
    embedding.getFloat(1L) shouldBe 0.2f +- 0.001f
    embedding.getFloat(2L) shouldBe 0.3f +- 0.001f
  }

  "createInputEmbeddings" should "correctly combine token and positional embeddings" in {
    val embeddingsPath = createTempFile(testEmbeddingsData)
    val embeddingsMap = SlidingWindowSpark.loadEmbeddings(spark, embeddingsPath)

    val inputTokenIds = Array(1, 2)
    val embeddingDim = 3
    val inputEmbeddings = SlidingWindowSpark.createInputEmbeddings(inputTokenIds, embeddingsMap, embeddingDim)

    // Verify dimensions
    inputEmbeddings.rows() shouldBe 2
    inputEmbeddings.columns() shouldBe embeddingDim

    // Values will be token embeddings + positional embeddings
    // We're mainly verifying the shape and that the operation completed successfully
    inputEmbeddings should not be null
  }

  // Clean up
  override def afterAll(): Unit = {
    super.afterAll()
    if (spark != null) {
      spark.stop()
    }
  }
}