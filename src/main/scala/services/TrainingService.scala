package services

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.conf.{MultiLayerConfiguration, NeuralNetConfiguration}
import org.deeplearning4j.nn.conf.layers.{LSTM, RnnOutputLayer}
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.deeplearning4j.spark.api.Repartition
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.nd4j.linalg.activations.Activation
import org.deeplearning4j.util.ModelSerializer

object TrainingService {

  def main(args: Array[String]): Unit = {
    // Set up Spark session and context
    val spark = SparkSession.builder()
      .appName("TrainingService")
      .master("local[*]") // Adjust master as needed
      .getOrCreate()

    val sc = spark.sparkContext

    // Define embedding dimension at top level
    val embeddingDim = 50 // Set this to your embeddings' actual dimension

    // Load the serialized RDD
    val serializableRDD: RDD[(Array[Byte], Array[Byte])] = sc.objectFile("src/main/resources/trainingData/part-00000")

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
        // If target is a single vector, reshape to [1, embeddingDim, 1]
        val target = targetEmbedding.reshape(1, embeddingDim, 1)
        // Tile the target to match the input sequence length
        Nd4j.tile(target, 1, 1, sequenceLength)
      }

      // Verify shapes
      println(s"Adjusted input shape: ${input.shape().mkString(",")}")
      println(s"Adjusted target shape: ${targetSequence.shape().mkString(",")}")

      // Create DataSet
      new DataSet(input, targetSequence)
    }

    // Define network hyperparameters
    val hiddenLayerSize = 128
    val batchSize = 32
    val numEpochs = 5

    // Configure the TrainingMaster for distributed training
    val tm = new ParameterAveragingTrainingMaster.Builder(batchSize)
      .averagingFrequency(5)
      .workerPrefetchNumBatches(2)
      .batchSizePerWorker(batchSize)
      .repartionData(Repartition.Always)
      .build()

    // Define the neural network configuration using LSTM
    val conf: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
      .seed(12345)
      .weightInit(WeightInit.XAVIER)
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

    // Create the SparkDl4jMultiLayer model
    val sparkModel = new SparkDl4jMultiLayer(sc, network, tm)

    // Train the model for the specified number of epochs
    for (epoch <- 1 to numEpochs) {
      sparkModel.fit(dataSetRDD)
      println(s"Completed epoch $epoch")
    }

    // Save the trained model
    ModelSerializer.writeModel(sparkModel.getNetwork, "src/main/resources/trainedModel.zip", true)
    println("Model saved at: src/main/resources/trainedModel.zip")

    // Stop Spark session
    spark.stop()
  }
}
