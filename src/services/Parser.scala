package services

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row, SQLContext, SparkSession}
import org.apache.spark.ml.feature.{StringIndexer, StringIndexerModel}
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}

/**
  * Created by james on 18/09/16.
  */
object Parser {

  /**
    * Parses the raw input to give Rating objects (user,item,preference)
    *
    */
  case class RawRow(user:String,artist:String,play_count:Double)
  def parseData(data_filepath:String,sc:SparkContext,sparks:SQLContext):(RDD[Rating],(StringIndexerModel,StringIndexerModel)) = {
    import sparks.implicits._
    val count_value_regex = "\b0*([1-9][0-9]*|0)\b".r
    val raw_data: DataFrame = sc.textFile(data_filepath).map {
      l => l.split("\t")
    }.map{
      case Array(user:String,artist:String,artist_name:String,count:String)
      => RawRow(user,artist,count.trim.toDouble)
      case x => throw new Exception(s"Unable to parse raw row $x")
    }.toDF()
    val user_string_indexer = new StringIndexer()
      .setInputCol("user")
      .setOutputCol("UserID")
      .fit(raw_data)
    val artist_string_indexer = new StringIndexer()
        .setInputCol("artist")
        .setOutputCol("ArtistID")
        .fit(raw_data)
    val indexed_data = user_string_indexer
      .transform(artist_string_indexer
        .transform(raw_data))
    val user_totals = indexed_data
      .select("UserID","play_count")
      .groupBy("UserID")
      .sum("play_count")
    val user_data_with_totals = indexed_data
      .join(user_totals,Seq("UserID"))
      .withColumn("RelativePreference",$"play_count"/$"sum(play_count)")
    val rating_rdd:RDD[Rating] = user_data_with_totals
      .select("UserID","ArtistID","RelativePreference")
      .rdd.map{
      case Row(userid:Double,artistid:Double,preference:Double)
      => Rating(userid.toInt,artistid.toInt,preference.toFloat)
      case x => throw new Exception(s"Unable to parse indexed row $x")
    }
    (rating_rdd,(user_string_indexer,artist_string_indexer))
  }

}
