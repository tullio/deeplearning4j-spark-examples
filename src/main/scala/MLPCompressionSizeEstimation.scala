import org.apache.spark.sql.SparkSession

object MLPCompressionSizeEstimation {

  def main(args: Array[String]): Unit = {
     val spark = SparkSession.builder()// .master("yarn")
       .getOrCreate()
     import spark.implicits._
    val sc = spark.sparkContext
    var tmMode = ""
    var trainingSamples = 0
    var testSamples = 0
    //parser.parse(args, Config()) match {
    ArgParser.parse(args) match {
      case Some(config) =>
        // do stuff
        tmMode = config.tmMode
        trainingSamples = config.trainingSamples
        testSamples = config.testSamples
      case None =>
        // arguments are bad, error message will have been displayed
        return
     }
  }
}
