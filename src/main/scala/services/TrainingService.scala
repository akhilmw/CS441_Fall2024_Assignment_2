package services

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.storage.StorageLevel
import org.deeplearning4j.nn.api.Model
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.conf.{MultiLayerConfiguration, NeuralNetConfiguration}
import org.deeplearning4j.nn.conf.layers.{LSTM, RnnOutputLayer}
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.api.TrainingListener
import org.deeplearning4j.optimize.api.BaseTrainingListener
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.deeplearning4j.spark.api.Repartition
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.learning.config.{Adam, Sgd}
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.nd4j.linalg.activations.Activation
import org.nd4j.evaluation.regression.RegressionEvaluation
import org.deeplearning4j.util.ModelSerializer
import org.deeplearning4j.spark.api.stats.SparkTrainingStats

import java.io.BufferedWriter
import scala.collection.mutable
import java.util
import java.time.{Duration, LocalDateTime}
import java.time.format.DateTimeFormatter
import java.util.concurrent.ConcurrentLinkedQueue

// Import logging
import org.slf4j.{Logger, LoggerFactory}

// Import Hadoop classes for HDFS operations
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import java.net.URI
import java.io.OutputStreamWriter
import java.io.OutputStream

// Custom training listener to collect detailed statistics
@SerialVersionUID(100L)
class CustomTrainingStatsListener extends BaseTrainingListener with Serializable {
  private var iterationCount = 0
  private var epochStartTime: Long = _
  private var trainingStartTime: Long = _
  private val stats = new ConcurrentLinkedQueue[String]()

  // Track metrics
  private var currentLearningRate = 0.0
  private var lastScore = 0.0
  private var lastParameters: INDArray = _
  private var gradientNorm: Double = 0.0
  private var parameterNorm: Double = 0.0
  private val logger = LoggerFactory.getLogger(getClass)

  override def iterationDone(model: Model, iteration: Int, epoch: Int): Unit = {
    if (trainingStartTime == 0) {
      trainingStartTime = System.currentTimeMillis()
    }

    iterationCount += 1
    lastScore = model.score()

    val network = model.asInstanceOf[MultiLayerNetwork]

    try {
      // Get learning rate
      currentLearningRate = network.getLayerWiseConfigurations
        .getConf(0)
        .getLayer
        .getUpdaterByParam("W")
        .asInstanceOf[Adam]
        .getLearningRate

      // Calculate gradient statistics
      val gradients = network.gradient().gradient()
      gradientNorm = gradients.norm2Number().doubleValue()

      val currentParams = network.params()
      parameterNorm = currentParams.norm2Number().doubleValue()

      val parameterUpdate = if (lastParameters != null) {
        currentParams.sub(lastParameters).norm2Number().doubleValue()
      } else {
        0.0
      }
      lastParameters = currentParams.dup()

      val currentTime = System.currentTimeMillis()
      val elapsedTime = Duration.ofMillis(currentTime - trainingStartTime)

      val runtime = Runtime.getRuntime
      val usedMemory = (runtime.totalMemory() - runtime.freeMemory()) / (1024 * 1024)
      val maxMemory = runtime.maxMemory() / (1024 * 1024)

      val statsLine = f"""
                         |=== Iteration ${iterationCount} Statistics ===
                         |Time: ${LocalDateTime.now.format(DateTimeFormatter.ISO_LOCAL_TIME)}
                         |Epoch: $epoch
                         |Training Metrics:
                         |  - Loss: $lastScore%.6f
                         |  - Learning Rate: $currentLearningRate%.6e
                         |  - Gradient Norm: $gradientNorm%.6f
                         |  - Parameter Norm: $parameterNorm%.6f
                         |  - Parameter Update: $parameterUpdate%.6f
                         |
                         |Memory Usage:
                         |  - Used Memory: ${usedMemory}MB
                         |  - Max Memory: ${maxMemory}MB
                         |  - Utilization: ${(usedMemory.toDouble / maxMemory.toDouble * 100).round}%%
                         |
                         |Time Metrics:
                         |  - Total Time: ${elapsedTime.toMinutes} minutes
                         |  - Iterations/sec: ${iterationCount.toDouble / (elapsedTime.toMillis / 1000.0)}%.2f
                         |----------------------------------------
        """.stripMargin

      stats.offer(statsLine)
      logger.info(statsLine)
    } catch {
      case e: Exception =>
        val errorMsg = s"Error collecting metrics: ${e.getMessage}"
        stats.offer(errorMsg)
        logger.error(errorMsg, e)
    }
  }

  override def onEpochStart(model: Model): Unit = {
    epochStartTime = System.currentTimeMillis()
    if (trainingStartTime == 0) trainingStartTime = epochStartTime

    val statsLine = f"""
                       |=== Epoch Start ===
                       |Time: ${LocalDateTime.now.format(DateTimeFormatter.ISO_LOCAL_TIME)}
                       |----------------------------------------
      """.stripMargin

    stats.offer(statsLine)
    logger.info(statsLine)
  }

  override def onEpochEnd(model: Model): Unit = {
    val epochDuration = System.currentTimeMillis() - epochStartTime
    val network = model.asInstanceOf[MultiLayerNetwork]

    // Get layer-specific information
    val layerInfo = new StringBuilder()
    for (i <- 0 until network.getnLayers) {
      val layer = network.getLayer(i)
      val layerConf = network.getLayerWiseConfigurations.getConf(i).getLayer

      // Access layer properties safely using pattern matching
      val (nIn, nOut) = layerConf match {
        case lstm: LSTM => (lstm.getNIn, lstm.getNOut)
        case rnn: RnnOutputLayer => (rnn.getNIn, rnn.getNOut)
        case _ => (0, 0) // Default case
      }

      val activation = layerConf match {
        case lstm: LSTM => lstm.getActivationFn
        case rnn: RnnOutputLayer => rnn.getActivationFn
        case _ => "Unknown"
      }

      layerInfo.append(f"""
                          |Layer $i (${layer.getClass.getSimpleName}):
                          |  - Parameters: ${layer.numParams()}
                          |  - Input/Output: $nIn/$nOut
                          |  - Activation: $activation
        """.stripMargin)
    }

    val statsLine = f"""
                       |=== Epoch End Statistics ===
                       |Duration: ${epochDuration}ms
                       |Final Metrics:
                       |  - Loss: $lastScore%.6f
                       |  - Learning Rate: $currentLearningRate%.6e
                       |  - Gradient Norm: $gradientNorm%.6f
                       |  - Parameter Norm: $parameterNorm%.6f
                       |
                       |Layer Information:
                       |$layerInfo
                       |----------------------------------------
      """.stripMargin

    stats.offer(statsLine)
    logger.info(statsLine)
  }

  def getStats: String = {
    val builder = new StringBuilder()
    val iterator = stats.iterator()
    while (iterator.hasNext) {
      builder.append(iterator.next())
      builder.append("\n")
    }
    builder.toString()
  }

  // Helper method to get current memory usage
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

object TrainingService {
  // Initialize logger
  private val log: Logger = LoggerFactory.getLogger(getClass)

  def main(args: Array[String]): Unit = {

    // for emr :
    if (args.length < 3) {
      System.err.println("Usage: TrainingService <trainingDataPath> <statsPath> <modelPath>")
      System.exit(1)
    }

    val trainingDataPath = args(0)
    val statsPath = args(1)
    val modelPath = args(2)



    // Set up Spark session and context
    val spark = SparkSession.builder()
      .appName("TrainingService")
//      .master("spark://Akhils-MacBook-Air.local:7077") // Spark master URL
      .config("spark.executor.memory", "4g")
      .config("spark.driver.memory", "2g")
      .config("spark.eventLog.enabled", "true")
//      .config("spark.eventLog.dir", "hdfs://localhost:9000/tmp/spark-events") // HDFS for event logs
      .config("spark.eventLog.dir", "hdfs:///user/akhil/spark-events") // HDFS relative path
//      .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000") // Use HDFS as default FS
      .getOrCreate()

    val sc = spark.sparkContext

    // Define embedding dimension at top level
    val embeddingDim = 50 // Set this to your embeddings' actual dimension

    // Initialize statistics collection
    val trainingStats = new StringBuilder()
    val startTime = LocalDateTime.now()

    // Load the serialized RDD from HDFS
//    val serializableRDD: RDD[(Array[Byte], Array[Byte])] = sc.objectFile("hdfs:///user/akhil/trainingData/part-00000")
    val serializableRDD: RDD[(Array[Byte], Array[Byte])] = sc.objectFile(trainingDataPath)

    // Cache the RDD to optimize performance
    serializableRDD.persist(StorageLevel.MEMORY_ONLY)

    // Convert RDD[(Array[Byte], Array[Byte])] to RDD[DataSet]
    val dataSetRDD: RDD[DataSet] = serializableRDD.map { case (inputBytes, targetBytes) =>
      val inputEmbeddings = Nd4j.fromByteArray(inputBytes)
      val targetEmbedding = Nd4j.fromByteArray(targetBytes)

      // Get shapes
      val inputShape = inputEmbeddings.shape()
      val sequenceLength = if (inputShape.length == 2) inputShape(0).toInt else 1

      // Reshape and permute input to [1, embeddingDim, sequenceLength]
      val input = if (sequenceLength > 1) {
        inputEmbeddings.reshape(1, sequenceLength, embeddingDim).permute(0, 2, 1)
      } else {
        inputEmbeddings.reshape(1, embeddingDim, 1)
      }

      // Handle target embedding
      val targetSequence = if (targetEmbedding.shape().length == 2) {
        targetEmbedding.reshape(1, targetEmbedding.shape()(0).toInt, embeddingDim).permute(0, 2, 1)
      } else {
        val target = targetEmbedding.reshape(1, embeddingDim, 1)
        Nd4j.tile(target, 1, 1, sequenceLength)
      }

      new DataSet(input, targetSequence)
    }

    // Split data into training and validation sets
    val Array(trainingData, validationData) = dataSetRDD.randomSplit(Array(0.8, 0.2))
    validationData.cache()

    // Define network hyperparameters
    val hiddenLayerSize = 128
    val batchSize = 32
    val numEpochs = 5

    val tm = new ParameterAveragingTrainingMaster.Builder(batchSize)
      .averagingFrequency(5)
      .batchSizePerWorker(batchSize)
      .workerPrefetchNumBatches(2)
      .collectTrainingStats(true)
      .exportDirectory("/user/akhil/input/spark")
      .build()

    // Define the neural network configuration using LSTM
    val conf: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
      .seed(12345)
      .weightInit(WeightInit.XAVIER)
      .updater(new Adam())
      .list()
      .layer(new LSTM.Builder()
        .nIn(embeddingDim)
        .nOut(hiddenLayerSize)
        .activation(Activation.TANH)
        .build())
      .layer(new RnnOutputLayer.Builder(LossFunction.MSE)
        .activation(Activation.IDENTITY)
        .nIn(hiddenLayerSize)
        .nOut(embeddingDim)
        .build())
      .build()

    // Initialize the network
    val network = new MultiLayerNetwork(conf)
    network.init()

    // Set up listeners for performance monitoring
    val statsListener = new CustomTrainingStatsListener()
    val listeners = new util.ArrayList[TrainingListener]()
    listeners.add(new ScoreIterationListener(10))
    listeners.add(statsListener)

    // Create the SparkDl4jMultiLayer model
    val sparkModel = new SparkDl4jMultiLayer(sc, network, tm)

    // Attach listeners to the sparkModel
    sparkModel.setListeners(listeners)

    // Variables to track epoch times
    val epochTimes = mutable.ArrayBuffer[Long]()

    // Train the model for the specified number of epochs
    for (epoch <- 1 to numEpochs) {
      val epochStartTime = System.currentTimeMillis()

      // Collect pre-training memory stats
      val preTrainingMemory = Runtime.getRuntime.totalMemory() - Runtime.getRuntime.freeMemory()
      val executorStats = sc.getExecutorMemoryStatus

      sparkModel.fit(trainingData)
      log.info(s"Completed epoch $epoch")

      val epochEndTime = System.currentTimeMillis()
      val epochDuration = epochEndTime - epochStartTime
      epochTimes += epochDuration

      // Collect post-training memory stats
      val postTrainingMemory = Runtime.getRuntime.totalMemory() - Runtime.getRuntime.freeMemory()

      log.info(s"Epoch $epoch time: $epochDuration ms")

      // Perform evaluation on validation data
      log.info(s"Performing evaluation for epoch $epoch")

      val validationDataList = validationData.collect()
      val evaluation = new RegressionEvaluation(embeddingDim)

      for (dataSet <- validationDataList) {
        val features = dataSet.getFeatures
        val labels = dataSet.getLabels
        val predictions = network.output(features, false)
        evaluation.eval(labels, predictions)
      }

      // Calculate metrics
      var totalMSE = 0.0
      var totalRMSE = 0.0
      var totalMAE = 0.0
      var totalR2 = 0.0
      val numColumns = embeddingDim

      for (i <- 0 until numColumns) {
        totalMSE += evaluation.meanSquaredError(i)
        totalRMSE += evaluation.rootMeanSquaredError(i)
        totalMAE += evaluation.meanAbsoluteError(i)
        totalR2 += evaluation.rSquared(i)
      }

      val avgMSE = totalMSE / numColumns
      val avgRMSE = totalRMSE / numColumns
      val avgMAE = totalMAE / numColumns
      val avgR2 = totalR2 / numColumns

      // Format and append epoch statistics
      val epochStats = f"""
                          |=== Epoch $epoch Statistics ===
                          |Time: ${LocalDateTime.now.format(DateTimeFormatter.ISO_LOCAL_TIME)}
                          |Duration: ${epochDuration}ms
                          |Memory Usage:
                          |  - Pre-training: ${preTrainingMemory / 1024 / 1024}MB
                          |  - Post-training: ${postTrainingMemory / 1024 / 1024}MB
                          |  - Delta: ${(postTrainingMemory - preTrainingMemory) / 1024 / 1024}MB
                          |
                          |Validation Metrics:
                          |  - MSE: $avgMSE%.4f
                          |  - RMSE: $avgRMSE%.4f
                          |  - MAE: $avgMAE%.4f
                          |  - R2: $avgR2%.4f
                          |
                          |Spark Metrics:
                          |  - Active Executors: ${sc.getExecutorMemoryStatus.size}
                          |  - Total Partitions: ${trainingData.getNumPartitions}
                          |============================================
        """.stripMargin

      trainingStats.append(epochStats)
      log.info(epochStats)
    }

    // Log average time per epoch
    val averageEpochTime = epochTimes.sum / epochTimes.length
    log.info(s"Average epoch time: $averageEpochTime ms")

    // Save training statistics to HDFS
    val fs = FileSystem.get(new URI("hdfs:///"), sc.hadoopConfiguration)
//    val statsPath = new Path("hdfs:///user/akhil/training_stats.txt")
      val statsPathObj = new Path(statsPath)
    val writer = new BufferedWriter(new OutputStreamWriter(fs.create(statsPathObj, true)))

    // Write summary information
    val summary = f"""
                     |=== Training Summary ===
                     |Start Time: ${startTime.format(DateTimeFormatter.ISO_LOCAL_DATE_TIME)}
                     |End Time: ${LocalDateTime.now.format(DateTimeFormatter.ISO_LOCAL_DATE_TIME)}
                     |Total Duration: ${Duration.between(startTime, LocalDateTime.now).toMinutes} minutes
                     |Number of Epochs: $numEpochs
                     |Batch Size: $batchSize
                     |Hidden Layer Size: $hiddenLayerSize
                     |Average Epoch Time: ${averageEpochTime}ms
                     |============================
                     |
                     |Detailed Statistics:
                     |""".stripMargin

    writer.write(summary)
    writer.write(trainingStats.toString())
    writer.write("\nDetailed Training Metrics:\n")
    writer.write(statsListener.getStats)
    writer.close()

    log.info(s"Training statistics saved to hdfs:///user/akhil/training_stats.txt")

    // Save the trained model to HDFS
    val modelPathObj = new Path(modelPath)
    val outputStream: OutputStream = fs.create(modelPathObj)
    ModelSerializer.writeModel(network, outputStream, true)
    outputStream.close()
    log.info(s"Model saved at: $modelPath")

    // Stop Spark session
    spark.stop()
  }
}