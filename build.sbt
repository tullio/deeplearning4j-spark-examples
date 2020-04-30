// The simplest possible sbt build file is just one line:

scalaVersion := "2.12.11"
//scalaVersion := "2.11.12" // for EMR
// That is, to create a valid sbt build, all you've got to do is define the
// version of Scala you'd like your project to use.

// ============================================================================

// Lines like the above defining `scalaVersion` are called "settings". Settings
// are key/value pairs. In the case of `scalaVersion`, the key is "scalaVersion"
// and the value is "2.13.1"

// It's possible to define many kinds of settings, such as:

name := "aws-dl4jtest"
organization := "persol.co.jp"
version := "1.0"

// Note, it's not required for you to define these three settings. These are
// mostly only necessary if you intend to publish your library's binaries on a
// place like Sonatype or Bintray.


// Want to use a published library in your project?
// You can define other libraries as dependencies in your build like this:
//libraryDependencies += "org.typelevel" %% "cats-core" % "2.0.0"
// https://mvnrepository.com/artifact/com.github.scopt/scopt
libraryDependencies += "com.github.scopt" %% "scopt" % "4.0.0-RC2"
libraryDependencies += "org.scalactic" %% "scalactic" % "3.1.1"
libraryDependencies += "org.scalatest" %% "scalatest" % "3.1.1" % "test"

libraryDependencies += "org.deeplearning4j" % "deeplearning4j-core" % "1.0.0-beta6"
libraryDependencies += "org.deeplearning4j" % "dl4j-spark-parameterserver_2.12" % "1.0.0-beta6"
libraryDependencies += "org.deeplearning4j" % "dl4j-spark_2.12" % "1.0.0-beta6"
// https://mvnrepository.com/artifact/org.deeplearning4j/deeplearning4j-zoo
libraryDependencies += "org.deeplearning4j" % "deeplearning4j-zoo" % "1.0.0-beta6"

libraryDependencies += "org.nd4j" % "nd4j-native-platform" % "1.0.0-beta6"
libraryDependencies += "org.apache.spark" %% "spark-yarn" % "2.4.4"
libraryDependencies += "org.apache.spark" %% "spark-core" % "2.4.4"
libraryDependencies += "org.apache.spark" %% "spark-sql" % "2.4.4"
// https://mvnrepository.com/artifact/io.netty/netty-all
//libraryDependencies += "io.netty" % "netty-all" % "4.1.48.Final"


// Here, `libraryDependencies` is a set of dependencies, and by using `+=`,
// we're adding the cats dependency to the set of dependencies that sbt will go
// and fetch when it starts up.
// Now, in any Scala file, you can import classes, objects, etc., from cats with
// a regular import.

// TIP: To find the "dependency" that you need to add to the
// `libraryDependencies` set, which in the above example looks like this:

// "org.typelevel" %% "cats-core" % "2.0.0"

// You can use Scaladex, an index of all known published Scala libraries. There,
// after you find the library you want, you can just copy/paste the dependency
// information that you need into your build file. For example, on the
// typelevel/cats Scaladex page,
// https://index.scala-lang.org/typelevel/cats, you can copy/paste the sbt
// dependency from the sbt box on the right-hand side of the screen.

// IMPORTANT NOTE: while build files look _kind of_ like regular Scala, it's
// important to note that syntax in *.sbt files doesn't always behave like
// regular Scala. For example, notice in this build file that it's not required
// to put our settings into an enclosing object or class. Always remember that
// sbt is a bit different, semantically, than vanilla Scala.

// ============================================================================

// Most moderately interesting Scala projects don't make use of the very simple
// build file style (called "bare style") used in this build.sbt file. Most
// intermediate Scala projects make use of so-called "multi-project" builds. A
// multi-project build makes it possible to have different folders which sbt can
// be configured differently for. That is, you may wish to have different
// dependencies or different testing frameworks defined for different parts of
// your codebase. Multi-project builds make this possible.

// Here's a quick glimpse of what a multi-project build looks like for this
// build, with only one "subproject" defined, called `root`:

// lazy val root = (project in file(".")).
//   settings(
//     inThisBuild(List(
//       organization := "ch.epfl.scala",
//       scalaVersion := "2.13.1"
//     )),
//     name := "hello-world"
//   )

// To learn more about multi-project builds, head over to the official sbt
// documentation at http://www.scala-sbt.org/documentation.html

assemblyMergeStrategy in assembly := {
  case PathList("javax", "servlet", xs @ _*)         => MergeStrategy.first
  case PathList("javax", "xml", xs @ _*)         => MergeStrategy.first
  case PathList("com", "fasterxml", xs @ _*)         => MergeStrategy.first
  case PathList("org", "slf4j", xs @ _*)         => MergeStrategy.first
  case PathList("org", "apache", xs @ _*) => MergeStrategy.last
  case PathList("com", "google", xs @ _*) => MergeStrategy.last
  case PathList(ps @ _*) if ps.last endsWith ".properties" => MergeStrategy.first
  case PathList(ps @ _*) if ps.last endsWith ".xml" => MergeStrategy.first
  case PathList(ps @ _*) if ps.last endsWith ".types" => MergeStrategy.first
  case PathList(ps @ _*) if ps.last endsWith ".class" => MergeStrategy.first
  case PathList(ps @ _*) if ps.last endsWith "epoll_x86_64.so" => MergeStrategy.first
  case "UnusedStubClass.class"  => MergeStrategy.first
  case "jetty-dir.css"                            => MergeStrategy.first

  case "application.conf"                            => MergeStrategy.concat
  case "unwanted.txt"                                => MergeStrategy.discard
  // Failed
  case "org.apache.hadoop.fs.FileSystem"                                => MergeStrategy.discard
    // Great!
  case PathList(p @ _*) if p.last == "org.apache.hadoop.fs.FileSystem" => MergeStrategy.discard
  case x =>
    val oldStrategy = (assemblyMergeStrategy in assembly).value
    oldStrategy(x)
}
