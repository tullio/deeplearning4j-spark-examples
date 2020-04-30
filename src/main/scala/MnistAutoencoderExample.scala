import org.apache.commons.lang3.tuple.ImmutablePair
import org.apache.commons.lang3.tuple.Pair
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.nd4j.linalg.learning.config.AdaGrad
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.api.IterationListener
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.evaluation.classification.Evaluation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions
import javax.swing._
import java.awt._
import java.awt.image.BufferedImage
import java.util._
import java.util

import scala.collection.JavaConversions._
object MnistAutoencoderExample {
  def main(args: Array[String]): Unit = {
    val conf = new NeuralNetConfiguration.Builder()
    .seed(12345)
    .weightInit(WeightInit.XAVIER)
    .updater(new AdaGrad(0.05))
    .activation(Activation.RELU)
    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
    .l2(0.0001)
    .list()
    .layer(0, new DenseLayer.Builder().nIn(784).nOut(250)
            .build())
    .layer(1, new DenseLayer.Builder().nIn(250).nOut(10)
            .build())
    .layer(2, new DenseLayer.Builder().nIn(10).nOut(250)
            .build())
    .layer(3, new OutputLayer.Builder().nIn(250).nOut(784)
            .lossFunction(LossFunctions.LossFunction.MSE)
            .build())
    .build()
    val miniBatchSize = 100
    var net = new MultiLayerNetwork(conf)
    net.setListeners(new ScoreIterationListener(10))
    //Load data and split into training and testing sets. 40000 train, 10000 test
    val iter = new MnistDataSetIterator(miniBatchSize,50000,false)

    val featuresTrain = new util.ArrayList[INDArray]
    val featuresTest = new util.ArrayList[INDArray]
    val labelsTest = new util.ArrayList[INDArray]

    val rand = new util.Random(12345)

    while(iter.hasNext()){
      val next = iter.next() // feature vector(bachsize, 28*28=784) and label vector(batchsize, 10)
      //println("next=",next.getFeatures(), next.getLabels())
      val split = next.splitTestAndTrain(80, rand)  //80/20 split (from miniBatch = 100)
      //println("split=",split)
      featuresTrain.add(split.getTrain().getFeatures()) // (80, 784)
      val dsTest = split.getTest() 
      featuresTest.add(dsTest.getFeatures()) // (20, 784)
      val indexes = Nd4j.argMax(dsTest.getLabels(),1) //Convert from one-hot representation -> index
      labelsTest.add(indexes)
    }
    // the "simple" way to do multiple epochs is to wrap fit() in a loop
    val nEpochs = 30
    val modelFile = "MnistAutoencoderExample.model"
    import java.io.File
    if((new File(modelFile)).exists()) {
      net = MultiLayerNetwork.load(new File(modelFile), true)
    } else {
      (1 to nEpochs).foreach{ epoch =>  
        featuresTrain.forEach( data => net.fit(data, data))
        println("Epoch " + epoch + " complete");
      }
      net.save(new File(modelFile), true)
    }
    //val eval = net.evaluate[Evaluation]()
    //Evaluate the model on the test data
    //Score each example in the test set separately
    //Compose a map that relates each digit to a list of (score, example) pairs
    //Then find N best and N worst scores per digit
    val listsByDigit = new util.HashMap[Integer, ArrayList[Pair[Double, INDArray]]]

    (0 to 9).foreach{ i => listsByDigit.put(i, new util.ArrayList[Pair[Double, INDArray]]) }

    println("featuresTest.size=", featuresTest.size)
    (0 to featuresTest.size-1).foreach{ i => // size = 500 = 50000/100
      val testData = featuresTest.get(i) // (20, 784)
      //println("testData = ", testData.shape)
      //println("testData.rows = ", testData.rows())
      //println("testData.columns = ", testData.columns)
        val labels = labelsTest.get(i) // index

        (0 to testData.rows-1).foreach{ j => // 20
          val digit = labels.getDouble(j.toLong).toInt
          // Input that is not a matrix;
          // expected matrix (rank 2), got rank 1 array with shape [784].
          // Missing preprocessor or wrong input type? (layer name: layer0, layer index: 0, layer type: DenseLayer)
          //  val example = testData.getRow(j)
          // val score = net.score(new DataSet(example, example))
          val example = testData.getRow(j) // feature vector (784)
          val examples = example.reshape(1, example.length) // add miniBatchsize
            val score = net.score((new DataSet(examples, examples)))
            // Add (score, example) pair to the appropriate list
            val digitAllPairs = listsByDigit.get(digit)
            digitAllPairs.add(new ImmutablePair[Double, INDArray](score, example))
        }
    }

    //Sort each list in the map by score
    val c = new Comparator[Pair[Double, INDArray]]() {
      override def compare(o1: Pair[Double, INDArray],
                           o2: Pair[Double, INDArray]): Int =
        java.lang.Double.compare(o1.getLeft, o2.getLeft)
    }

    listsByDigit.values().forEach(digitAllPairs => Collections.sort(digitAllPairs, c)) // sort by score part

    //After sorting, select N best and N worst scores (by reconstruction error) for each digit, where N=5
    val best = new util.ArrayList[INDArray](50)
    val worst = new util.ArrayList[INDArray](50)

    (0 to 9).foreach{ i => 
        val list = listsByDigit.get(i)

      (0 to 4).foreach{ j=>
           // getRight extract feature vector part
            best.add(list.get(j).getRight) 
            worst.add(list.get(list.size - j - 1).getRight)
        }
    }
    println("50 best") // examples that can be understandable will be shown 
    best.foreach{f => println(f)
    }
    println("50 worst")
    worst.foreach{f => println(f)
    }
  }
}
