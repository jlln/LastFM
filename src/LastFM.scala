/**
  * Created by james on 18/09/16.
  */

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.log4j.PropertyConfigurator

object LastFM {
  def main(args: Array[String]): Unit = {
    PropertyConfigurator.configure("conf/log4j.properties")
    val conf = new SparkConf().setAppName("LastFM").setMaster("local[3]")
    val sc = new SparkContext(conf)
    val sparks = new org.apache.spark.sql.SQLContext(sc)
    val input = "data/sample_user_artist.tsv"
    val rating_rdd_and_indexers = services.Parser.parseData(input,sc,sparks)
    val rating_rdd = rating_rdd_and_indexers._1
    rating_rdd.cache()
    val indexers = rating_rdd_and_indexers._2
    val user_indexer = indexers._1
    val artist_indexer = indexers._2
    services.MFModelling.tune(rating_rdd,sc)
  }
}
