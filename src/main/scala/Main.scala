import services.TrainingService
import services.SlidingWindowSpark
object Main {
  def main(args: Array[String]): Unit = {

    SlidingWindowSpark.main(Array("true"))
    TrainingService.main(Array("true"))

  }
}



