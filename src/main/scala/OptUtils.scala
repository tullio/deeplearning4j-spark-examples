import scopt.OptionParser

case class Config(
  tmMode: String = "ParameterAveraging",
  trainingSamples: Int = 1000,
  testSamples: Int = 200
)

object ArgParser {
  val parser = new scopt.OptionParser[Config]("scopt") {
    head("scopt", "3.x")
    opt[String]('t', "tm")
      .action((x, c) => c.copy(tmMode = x))
      .text("ParameterAveraging|GradientSharing")
    opt[Int]('r', "traingSamples")
      .required()
      .valueName("<sample number>")
      .action((x, c) => c.copy(trainingSamples = x))
      .text("Number of training samples")
    opt[Int]('e', "testSamples")
      .required()
      .valueName("<sample number>")
      .action((x, c) => c.copy(testSamples = x))
      .text("Number of test samples")
  }
  def parse(args: Array[String]) = {
    parser.parse(args, Config())
  }
}
