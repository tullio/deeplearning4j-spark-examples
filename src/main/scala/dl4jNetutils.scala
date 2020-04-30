import java.net.NetworkInterface
import collection.JavaConverters._
import java.util.Collections
import java.util.ArrayList

object Netutils {
  def getNicList(nics: ArrayList[NetworkInterface]) = {
    nics.asScala.foldLeft(List.empty[NetworkInterface])((f, g) => f :+ g)
  }
  def getNetmask(netmask: Int) = {
    val binMask = (0xffffffff << (32-netmask))
    println(s"binMask = ${Integer.toBinaryString(binMask)}")
    binMask
  }
  def getIPv4string(decimalNetworkAddress: Long) = {
    val stringNetworkAddress = Seq(0,0,0,0).foldLeft((decimalNetworkAddress, ""))((f, g) => (f._1 / 256, (f._1 % 256).toString + "." + f._2))._2
    val networkAddress = stringNetworkAddress.substring(0, stringNetworkAddress.length - 1)
    networkAddress
  }
  def getDecimalAddress(byteArray: Array[Byte]) = {
    val sdecimalAddress = byteArray.map(f => if(f<0) 256+f else f).foldLeft(0)((f,g) => (f << 8) + g)
    val decimalAddress = if(sdecimalAddress > 0) sdecimalAddress else 65536L*65536+sdecimalAddress
    decimalAddress
  }
  def getNetworkInfoString(nicList: List[NetworkInterface]) = {
    // List[scala.collection.mutable.Buffer[(String, Int)]]
    // = List(ArrayBuffer((fe80:0:0:0:81b:69ff:fe4f:21c4%eth0,64), (10.1.1.224,24)), ArrayBuffer((0:0:0:0:0:0:0:1%lo,128), (127.0.0.1,8)))
    val inetPairs =  nicList.map(f => f.getInterfaceAddresses.asScala.map(g => (g.getAddress, g.getNetworkPrefixLength.toInt)))
    println(s"inetPairs=${inetPairs}")
    // (String, Int) = (10.1.1.224,24)
    val ipv4tupple =  inetPairs.filter(f => f(0)._2 == 64)(0)(1) // should filter by name(eth0?) instead of 64?
    println(s"ipv4tupple=${ipv4tupple}")
    val addressString = ipv4tupple._1.getHostAddress()
    val netmask = ipv4tupple._2
    val decimalAddress = Netutils.getDecimalAddress(ipv4tupple._1.getAddress())
    println(s"decimalAddress = ${decimalAddress}")
    //println(s"decimalAddress = ${Integer.toBinaryString(decimalAddress)}")
    println(s"decimalAddress = ${java.lang.Long.toBinaryString(decimalAddress)}")
    val binMask = Netutils.getNetmask(netmask)
    val decimalNetworkAddress = decimalAddress & binMask
    val networkAddress = Netutils.getIPv4string(decimalNetworkAddress)
    val networkInfo = s"${networkAddress}/${netmask}"
    println(s"netmask in driver = ${networkAddress}/${netmask}")
    println(s"address in driver = ${addressString}")
    (addressString, networkInfo)
  }
}
class Netutils() {
  val nics = Collections.list(NetworkInterface.getNetworkInterfaces)
  // List[java.net.NetworkInterface] = List(name:eth0 (eth0), name:lo (lo))
  val nicList = Netutils.getNicList(nics)
  println(s"nicList=${nicList}")
  val networkInfo = Netutils.getNetworkInfoString(nicList)
  val host = networkInfo._1
  val network = networkInfo._2
  def getHost() = {
    host
  }
  def getNetwork() = {
    network
  }
}
