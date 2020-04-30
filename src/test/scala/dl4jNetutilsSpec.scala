import org.scalatest._
import java.net.NetworkInterface
import java.util.Collections
import collection.JavaConverters._

class dl4jNetutilsSpecSpec extends FlatSpec with Matchers {
  "Network Information" should "be calculated" in {
    val nics = Collections.list(NetworkInterface.getNetworkInterfaces)
    val nicList = Netutils.getNicList(nics)
    //nicList.foreach(f => println(f.getName(), f.getDisplayName(), f.getInetAddresses(), f.getHardwareAddress(), f.getInterfaceAddresses()))
    assert(nicList.filter(f => f.getName().contains("eth")).length > 0)
    assert(nicList.filter(f => f.getName().contains("lo")).length > 0)
    val networkInfoString = Netutils.getNetworkInfoString(nicList)
    println(networkInfoString)
  }
  "Network mask bit series" should "be obtained" in {
    assert(Integer.toBinaryString(Netutils.getNetmask(16)) == "11111111111111110000000000000000")
    assert(Integer.toBinaryString(Netutils.getNetmask(1)) == "10000000000000000000000000000000")
    assert(Integer.toBinaryString(Netutils.getNetmask(32)) == "11111111111111111111111111111111")
  }
  "Dot-decimal notation string" should "be obtained" in {
    assert(Netutils.getIPv4string(2130706433L) == "127.0.0.1")
    assert(Netutils.getIPv4string(3232235777L) == "192.168.1.1")
  }
  "Decimal address" should "be obtained" in {
    assert(Netutils.getDecimalAddress(Array(127, 0, 0, 1)) == 2130706433L)
    assert(Netutils.getDecimalAddress(Array(192.toByte, 168.toByte, 1, 1)) == 3232235777L)
  }
}
