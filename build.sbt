// See README.md for license details.

//ThisBuild / scalaVersion     := "2.13.16"
ThisBuild / scalaVersion := "2.13.6"
ThisBuild / version          := "0.1.0"

//val chiselVersion = "7.0.0"
val chiselVersion = "6.5.0"

lazy val root = (project in file("."))
  .settings(
    name := "mnist-chisel-online",
    libraryDependencies ++= Seq(
      "org.chipsalliance" %% "chisel" % chiselVersion,
      //"edu.berkeley.cs" %% "chisel3"     % "7.0.0",
      //"org.scalatest" %% "scalatest" % "3.2.19" % "test",
      "edu.berkeley.cs" %% "fixedpoint" % "1.0.0"
    ),
    //  libraryDependencies ++= Seq(
    //   "edu.berkeley.cs" %% "chisel3"     % "6.5.0",
    //   "edu.berkeley.cs" %% "chiseltest"  % "6.0.0" % Test
    // ),

    scalacOptions ++= Seq(
      "-language:reflectiveCalls",
      "-deprecation",
      "-feature",
      "-Xcheckinit",
      "-Ymacro-annotations",
    ),
    addCompilerPlugin("org.chipsalliance" % "chisel-plugin" % chiselVersion cross CrossVersion.full),
    // テストを並列化しない（波形・状態が絡むため）
    Test / parallelExecution := false
  )

