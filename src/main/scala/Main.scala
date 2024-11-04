import services.TrainingService
import services.SlidingWindowSpark
object Main {

  /**
   * Main entry point of the application
   *
   * @param args Command line arguments (not used in this implementation)
   *
   * The method executes the pipeline in two stages:
   * 1. First runs the SlidingWindowSpark to prepare the training data
   * 2. Then executes the TrainingService to train the neural network
   *
   * Both services are initialized in local mode (true parameter)
   * Setting the parameter to false would run them in cluster mode
   */

  def main(args: Array[String]): Unit = {

    // Execute sliding window preprocessing
    // Parameter "true" indicates local mode execution
    SlidingWindowSpark.main(Array("true"))

    // Execute neural network training
    // Parameter "true" indicates local mode execution
    TrainingService.main(Array("true"))

  }
}



