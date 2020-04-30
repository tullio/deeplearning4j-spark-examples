import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.spark.parameterserver.training.SharedTrainingMaster
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster

import org.nd4j.evaluation.classification.Evaluation
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.Nadam
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration
import org.nd4j.parameterserver.distributed.v2.enums.MeshBuildMode
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.apache.spark.sql.SparkSession
import org.nd4j.linalg.dataset.DataSet
import collection.JavaConverters._



/** A slightly more involved multilayered (MLP) applied to digit classification for the MNIST dataset (http://yann.lecun.com/exdb/mnist/).
 *
 * This example uses two input layers and one hidden layer.
 *
 * The first input layer has input dimension of numRows*numColumns where these variables indicate the
 * number of vertical and horizontal pixels in the image. This layer uses a rectified linear unit
 * (relu) activation function. The weights for this layer are initialized by using Xavier initialization
 * (https://prateekvjoshi.com/2016/03/29/understanding-xavier-initialization-in-deep-neural-networks/)
 * to avoid having a steep learning curve. This layer sends 500 output signals to the second layer.
 *
 * The second input layer has input dimension of 500. This layer also uses a rectified linear unit
 * (relu) activation function. The weights for this layer are also initialized by using Xavier initialization
 * (https://prateekvjoshi.com/2016/03/29/understanding-xavier-initialization-in-deep-neural-networks/)
 * to avoid having a steep learning curve. This layer sends 100 output signals to the hidden layer.
 *
 * The hidden layer has input dimensions of 100. These are fed from the second input layer. The weights
 * for this layer is also initialized using Xavier initialization. The activation function for this
 * layer is a softmax, which normalizes all the 10 outputs such that the normalized sums
 * add up to 1. The highest of these normalized values is picked as the predicted class.
 *
 */

object SparkMLPMnistTwoLayerExample {
  //private val log = LoggerFactory.getLogger(classOf[MLPMnistTwoLayerExample])

          //private val log = LoggerFactory.getLogger(classOf[MLPMnistTwoLayerExample])

      //@throws[Exception]
  def main(args: Array[String]): Unit = {
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
     val spark = SparkSession.builder()// .master("yarn")
       .getOrCreate()
     import spark.implicits._
     //number of rows and columns in the input pictures
     val numRows = 28
     val numColumns = 28
     val outputNum = 10 // number of output classes
     val batchSize = 16 // batch size for each epoch
     val rngSeed = 123 // random number seed for reproducibility
     val numEpochs = 15 // number of epochs to perform
     val rate = 0.0015 // learning rate
     //Get the DataSetIterators:
     val mnistTrain = new MnistDataSetIterator(batchSize, true, rngSeed)
     val mnistTest = new MnistDataSetIterator(batchSize, false, rngSeed)
     println("Build model....")
     //val conf = (new NeuralNetConfiguration.Builder()).seed(rngSeed) //include a random seed for reproducibility
     val conf = (new NeuralNetConfiguration.Builder()).seed(rngSeed) //include a random seed for reproducibility
          .activation(Activation.RELU)
          .weightInit(WeightInit.XAVIER)
          .updater(new Nadam())
          .l2(rate * 0.005).list // regularize learning model
          .layer((new DenseLayer.Builder()) //create the first input layer.
                  .nIn(numRows * numColumns)
                  .nOut(500).build())
          .layer((new DenseLayer.Builder())
            .nIn(500) //create the second input layer
            .nOut(100).build())
          .layer((new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD))
            .activation(Activation.SOFTMAX) //create hidden layer
            .nOut(outputNum).build())
          .build()

            import org.deeplearning4j.optimize.solvers.accumulation.encoding.threshold.AdaptiveThresholdAlgorithm
            import org.nd4j.parameterserver.distributed.v2.enums.MeshBuildMode

        import java.net.NetworkInterface
        val nics = NetworkInterface.getNetworkInterfaces
        /**
        while(nics.hasMoreElements()){
          val nic = nics.nextElement
          val inetPairs =  nic.getInterfaceAddresses.asScala
          .map(f => (f.getAddress.getHostAddress, f.getNetworkPrefixLength.toInt))
          println(s"inetPairs=${inetPairs}")
        }
        return
          **/
        // List[java.net.NetworkInterface] = List(name:eth0 (eth0), name:lo (lo))
        val nicList = nics.asScala.foldLeft(List.empty[java.net.NetworkInterface])((f, g) => f :+ g)
        println(s"nicList=${nicList}")
        // List[scala.collection.mutable.Buffer[(String, Int)]]
        // = List(ArrayBuffer((fe80:0:0:0:81b:69ff:fe4f:21c4%eth0,64), (10.1.1.224,24)), ArrayBuffer((0:0:0:0:0:0:0:1%lo,128), (127.0.0.1,8)))
        val inetPairs =  nicList.map(f => f.getInterfaceAddresses.asScala.map(g => (g.getAddress, g.getNetworkPrefixLength.toInt)))
        println(s"inetPairs=${inetPairs}")
        // (String, Int) = (10.1.1.224,24)
        val ipv4tupple =  inetPairs.filter(f => f(0)._2 == 64)(0)(1)
        println(s"ipv4tupple=${ipv4tupple}")
            //val nic = NetworkInterface.getNetworkInterfaces.nextElement().getInterfaceAddresses.get(0)
            //val address = nic.getAddress.getHostAddress
            //val netmask = nic.getNetworkPrefixLength // /(24)
        val addressString = ipv4tupple._1.getHostAddress()
        val netmask = ipv4tupple._2
        val sdecimalAddress = ipv4tupple._1.getAddress().map(f => if(f<0) 256+f else f).foldLeft(0)((f,g) => (f << 8) + g)
        val decimalAddress = if(sdecimalAddress > 0) sdecimalAddress else 65536L*65536+sdecimalAddress
        println(s"decimalAddress = ${decimalAddress}")
        //println(s"decimalAddress = ${Integer.toBinaryString(decimalAddress)}")
        println(s"decimalAddress = ${java.lang.Long.toBinaryString(decimalAddress)}")
        val binMask = (0xffffffff << (32-netmask))
        println(s"binMask = ${Integer.toBinaryString(binMask)}")
        val decimalNetworkAddress = decimalAddress & binMask
        val stringNetworkAddress = Seq(0,0,0,0).foldLeft((decimalNetworkAddress, ""))((f, g) => (f._1 / 256, (f._1 % 256).toString + "." + f._2))._2
        val networkAddress = stringNetworkAddress.substring(0, stringNetworkAddress.length - 1)
        println(s"netmask in driver = ${networkAddress}/${netmask}")
        println(s"address in driver = ${addressString}")
            //Set up TrainingMaster for gradient sharing training
            val voidConfiguration = VoidConfiguration.builder.unicastPort(45678)// Should be open for IN/OUT communications on all Spark nodes
              .networkMask(s"${networkAddress}/${netmask}") // Local network mask - for example, 10.0.0.0/16 - see https://deeplearning4j.org/docs/latest/deeplearning4j-scaleout-parameter-server
               .controllerAddress(addressString) // IP address of the master/driver node
             .meshBuildMode(MeshBuildMode.PLAIN) 
              .build()
        Seq(0,1,2, 3).toDS().map{f =>
          println(s"################ Network check ################## ID=${f}")
          val nics = NetworkInterface.getNetworkInterfaces
          val nicList = nics.asScala.foldLeft(List.empty[java.net.NetworkInterface])((f, g) => f :+ g)
          println(s"nicList=${nicList}")
          val inetPairs =  nicList.map(f =>
            f.getInterfaceAddresses.asScala.map(g =>
                  (g.getAddress, g.getNetworkPrefixLength.toInt)))
          println(s"inetPairs=${inetPairs}")
          val ipv4tupple =  inetPairs.filter(f => f(0)._2 == 64)(0)(1)
          println(s"ipv4tupple=${ipv4tupple}")
          val addressString = ipv4tupple._1.getHostAddress()
          val netmask = ipv4tupple._2
          val sdecimalAddress
          = ipv4tupple._1.getAddress().map(f =>
            if(f<0) 256+f else f).foldLeft(0)((f,g) => (f << 8) + g)
          val decimalAddress
            = if(sdecimalAddress > 0) sdecimalAddress else 65536L*65536+sdecimalAddress
          val binMask = (0xffffffff << (32-netmask))
          println(s"binMask = ${Integer.toBinaryString(binMask)}")
          val decimalNetworkAddress = decimalAddress & binMask
          val stringNetworkAddress
          = Seq(0,0,0,0).foldLeft((decimalNetworkAddress, "")) ((f, g) => (f._1 / 256, (f._1 % 256).toString + "." + f._2))._2
          val networkAddress
            = stringNetworkAddress.substring(0, stringNetworkAddress.length - 1)
          println(s"netmask in worker = ${networkAddress}/${netmask}")
          println(s"address in worker = ${addressString}")
          f
        }.collect()

    val minibatch = batchSize

        println(s"Training Master mode: ${tmMode}")
        val tm =
          if(tmMode == "ParameterAveraging") {
            new ParameterAveragingTrainingMaster.Builder(minibatch)
	      .build()
          } else if(tmMode == "GradientSharing") {
             new SharedTrainingMaster.Builder(voidConfiguration, minibatch)
                            .rngSeed(12345)
              .collectTrainingStats(true)
              .batchSizePerWorker(minibatch)
              .thresholdAlgorithm(new AdaptiveThresholdAlgorithm(1e-3))//Threshold algorithm determines the encoding threshold to be use. See docs for details
               .workersPerNode(1) // Workers per node
//               .meshBuildMode(MeshBuildMode.PLAIN)
                 .build() 
          } else {
            println(s"${tmMode} is not supported")
            null
          }

            //val model = new MultiLayerNetwork(conf)
            val model = new SparkDl4jMultiLayer(spark.sparkContext, conf, tm)
//        model.init
        model.setListeners(new ScoreIterationListener(1)) //print the score with every iteration
        val mnistTrainBuffer = scala.collection.mutable.ListBuffer.empty[DataSet]
        var i = 0
        while(mnistTrain.hasNext  && i < trainingSamples ) {
          val dataset = mnistTrain.next(1)
          //println(dataset)
              mnistTrainBuffer.append(dataset)
              i += 1
            }
            println(s"train data size=${mnistTrainBuffer.length}")
        val mnistTestBuffer = scala.collection.mutable.ListBuffer.empty[DataSet]
        i = 0
            while(mnistTest.hasNext && i < testSamples) {
              mnistTestBuffer.append(mnistTest.next(1))
              i += 1
            }
            println(s"test data size=${mnistTestBuffer.length}")
        val mnistTrainRDD = spark.sparkContext.makeRDD(mnistTrainBuffer)
        println(s"train RDD data size=${mnistTrainRDD.count()}")
            val mnistTestRDD = spark.sparkContext.makeRDD(mnistTestBuffer)
        println("Train model....")
            for(i <- 0 to 10){
              //model.fit(mnistTrain, numEpochs)
              model.fit(mnistTrainRDD)
            }

        println("Evaluate model....")
        //val eval = model.evaluate[Evaluation](mnistTest) // Evaluationを入れる
        val eval = model.evaluate[Evaluation](mnistTestRDD) // Evaluationを入れる
        println(eval.stats)
        println("****************Example finished********************")
      }
}
