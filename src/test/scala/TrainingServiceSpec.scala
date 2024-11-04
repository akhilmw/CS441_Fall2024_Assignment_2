package services

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.apache.spark.sql.SparkSession
import org.nd4j.linalg.factory.Nd4j
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.dataset.DataSet
import org.apache.spark.rdd.RDD
import java.nio.file.{Files, Path => JPath}
import java.time.LocalDateTime
import org.deeplearning4j.nn.api.Model
import java.io.File
import org.nd4j.linalg.learning.config.{Adam, Sgd}
import com.typesafe.config.ConfigFactory

class TrainingServiceSpec extends AnyFlatSpec with Matchers {

  // Test 1: Configuration values test
  "TrainingService" should "handle local and cluster paths correctly" in {
    val testConfig = ConfigFactory.load()
    testConfig.hasPath("filePaths.trainingDataPath") shouldBe true
    testConfig.hasPath("filePaths.statsPath") shouldBe true
    testConfig.hasPath("filePaths.modelPath") shouldBe true
  }

  // Test 2: CustomTrainingStatsListener test with proper configuration
  "CustomTrainingStatsListener" should "handle statistics collection correctly" in {
    val listener = new CustomTrainingStatsListener()
    val network = createTestNetwork(true) // Using Adam optimizer

    listener.onEpochStart(network)
    listener.onEpochEnd(network)

    val stats = listener.getStats
    stats should include("=== Epoch Start ===")
    stats should include("=== Epoch End Statistics ===")
    stats should include("Layer Information")
  }


  // Test 3: Layer parameter test
  it should "record correct number of parameters" in {
    val listener = new CustomTrainingStatsListener()
    val network = createTestNetwork(true)

    listener.onEpochEnd(network)

    val stats = listener.getStats
    stats should include("Parameters: 2550") // 50 * 50 + 50 biases = 2550 params per layer
  }

  // Test 4: Training metrics test
  it should "record training metrics" in {
    val listener = new CustomTrainingStatsListener()
    val network = createTestNetwork(true)

    listener.onEpochStart(network)
    listener.onEpochEnd(network)

    val stats = listener.getStats
    stats should include("Final Metrics")
    stats should include("Loss")
    stats should include("Gradient Norm")
  }

  // Helper method to create test network
  private def createTestNetwork(useAdam: Boolean = true): MultiLayerNetwork = {
    val conf = new org.deeplearning4j.nn.conf.NeuralNetConfiguration.Builder()
      .seed(12345)
      .updater(if (useAdam) new Adam(0.01) else new Sgd(0.01))
      .list()
      .layer(0, new org.deeplearning4j.nn.conf.layers.DenseLayer.Builder()
        .nIn(50)
        .nOut(50)
        .build())
      .layer(1, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder()
        .nIn(50)
        .nOut(50)
        .build())
      .build()

    val network = new MultiLayerNetwork(conf)
    network.init()
    network
  }

  // Helper method to get runtime memory stats
  private def getMemoryStats: String = {
    val runtime = Runtime.getRuntime
    val usedMemory = (runtime.totalMemory() - runtime.freeMemory()) / (1024 * 1024)
    val maxMemory = runtime.maxMemory() / (1024 * 1024)
    f"""Memory Usage:
       |  - Used Memory: ${usedMemory}MB
       |  - Max Memory: ${maxMemory}MB
       |  - Utilization: ${(usedMemory.toDouble / maxMemory.toDouble * 100).round}%%""".stripMargin
  }
}