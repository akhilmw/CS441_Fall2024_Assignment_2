package utils

import scala.io.Source
import scala.collection.mutable.ListBuffer
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.api.ndarray.INDArray
import com.typesafe.config.ConfigFactory

object SlidingWindowWithEmbeddings {

  private val config = ConfigFactory.load()


  def main(args: Array[String]) : Unit = {

    val embeddingsFilePath = config.getString("filePaths.embeddingsPath")
    val tokensFilePath = config.getString("filePaths.tokensPath")
    val windowSize = 4
    val embeddingDim = 50
    val embeddingsMap: Map[Int, INDArray] = loadEmbeddings(embeddingsFilePath)
    val tokenIds: Array[Int] = loadTokens(tokensFilePath)

    // Create sliding windows
    val slidingWindows = createSlidingWindows(tokenIds, windowSize)

    // Prepare training data
    val trainingData = slidingWindows.map { case (inputTokenIds, targetTokenId) =>
      val inputEmbeddings = createInputEmbeddings(inputTokenIds, embeddingsMap, embeddingDim)
      val targetEmbedding = getTokenEmbedding(targetTokenId, embeddingsMap, embeddingDim)
      (inputEmbeddings, targetEmbedding)
    }

    // Test the implementation
    trainingData.take(3).foreach { case (inputEmbeddings, targetEmbedding) =>
      println("Input Embeddings Shape: " + inputEmbeddings.shapeInfoToString())
      println("Target Embedding Shape: " + targetEmbedding.shapeInfoToString())
      println("Input Embeddings: " + inputEmbeddings)
      println("Target Embedding: " + targetEmbedding)
    }

  }

  // Function to load embeddings from embeddings.csv
  def loadEmbeddings(filePath: String): Map[Int, INDArray] = {
    val embeddingsMap = Source.fromFile(filePath).getLines().drop(1).map { line =>
      val parts = line.split(",")
      val tokenId = parts(0).toInt
      // The embeddings start from index 2 onwards
      val embeddingsArray = parts.slice(2, parts.length).map(_.toDouble)
      val embeddingVector = Nd4j.create(embeddingsArray)
      (tokenId, embeddingVector)
    }.toMap
    embeddingsMap
  }


  // Function to load token IDs from ordered_tokens.csv
  def loadTokens(filePath: String): Array[Int] = {
    val tokenIds = Source.fromFile(filePath).getLines().drop(1).flatMap { line =>
      val parts = line.split(",") // Changed delimiter to comma
      if (parts.length > 2) {
        val encodedTokensStr = parts(2)
        // Remove brackets and split
        val tokenIdStrs = encodedTokensStr.replace("[", "").replace("]", "").split("\\s+")
        tokenIdStrs.filter(_.nonEmpty).map(_.toInt)
      } else {
        println(s"Skipping malformed line: $line")
        Array.empty[Int]
      }
    }.toArray
    tokenIds
  }



  // Function to create sliding windows over token IDs
  private def createSlidingWindows(tokenIds: Array[Int], windowSize: Int): List[(Array[Int], Int)] = {
    val slidingWindows = new ListBuffer[(Array[Int], Int)]()

    for (i <- 0 until tokenIds.length - windowSize) {
      val inputSequence = tokenIds.slice(i, i + windowSize)
      val targetTokenId = tokenIds(i + windowSize)
      slidingWindows += ((inputSequence, targetTokenId))
    }

    slidingWindows.toList
  }


  // Function to create input embeddings with positional information
  def createInputEmbeddings(
                             inputTokenIds: Array[Int],
                             embeddingsMap: Map[Int, INDArray],
                             embeddingDim: Int
                           ): INDArray = {
    val sequenceLength = inputTokenIds.length
    val inputEmbeddings = Nd4j.create(sequenceLength, embeddingDim)

    for (i <- inputTokenIds.indices) {
      val tokenEmbedding = getTokenEmbedding(inputTokenIds(i), embeddingsMap, embeddingDim)
      val positionalEmbedding = computePositionalEmbedding(i, embeddingDim)
      val combinedEmbedding = tokenEmbedding.add(positionalEmbedding)
      inputEmbeddings.putRow(i, combinedEmbedding)
    }

    inputEmbeddings
  }

  // Function to get the embedding vector for a token ID
  def getTokenEmbedding(tokenId: Int, embeddingsMap: Map[Int, INDArray], embeddingDim: Int): INDArray = {
    embeddingsMap.getOrElse(tokenId, Nd4j.zeros(embeddingDim))
  }

  // Function to compute positional embedding for a given position
  private def computePositionalEmbedding(position: Int, dModel: Int): INDArray = {
    val embedding = Nd4j.create(dModel)
    for (i <- 0 until dModel) {
      val angle = position.toDouble / math.pow(10000, (2 * (i / 2)) / dModel.toDouble)
      val value = if (i % 2 == 0) math.sin(angle) else math.cos(angle)
      embedding.putScalar(i, value)
    }
    embedding
  }






}
