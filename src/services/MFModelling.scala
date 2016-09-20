package services

import java.io.{File, PrintWriter}

import org.apache.spark.SparkContext
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.rdd.RDD


/**
  * Created by james on 18/09/16.
  */
object MFModelling {
  def getRMSE(model: MatrixFactorizationModel, data: RDD[Rating], n: Long,max:Double,min:Double): Double = {
    val predictions: RDD[Rating] = model.predict(data.map(x => (x.user, x.product)))
    val predictionsAndRatings = predictions.map(x => ((x.user, x.product), x.rating))
      .join(data.map(x => ((x.user, x.product), x.rating))).values
    (math.sqrt(predictionsAndRatings.map(x => (x._1 - x._2) * (x._1 - x._2)).reduce(_ + _) / n))/(max-min)
  }

  def tune(ratings:RDD[Rating],sc:SparkContext) = {
    val output_log = new File("output_log.txt")
    val log_writer = new PrintWriter(output_log)
    val n_cases = ratings.count()
    val partitions = ratings.randomSplit(Array(0.6,0.2,0.2),1234L)
    val train = partitions(0)
    val valid = partitions(1)
    val num_valid = valid.count()
    val test = partitions(2)
    val alphas = List(10d,40d,60d)
    val lambdas = List(1d,10d)
    val ranks = List(8,16,32)
    var best_lambda = 0d
    var best_alpha = 0d
    var best_rank = 0
    var best_score = 100000d
    val preferences = ratings.map(r => r.rating)
    val max_pref = preferences.max()
    val min_pref = preferences.min()
    for (alpha<-alphas ; lambda<-lambdas;rank <-ranks){
      val model = ALS.trainImplicit(train,rank,10,lambda,alpha)
      val model_rmse = getRMSE(model,valid,num_valid,max_pref,min_pref)
      if (model_rmse < best_score){
        best_lambda = lambda
        best_alpha = alpha
        best_rank = rank
        best_score = model_rmse
      }
      log_writer.write(s"Lambda: $lambda Alpha: $alpha Rank: $rank Produced RMSE of $model_rmse \n")
      log_writer.flush()
    }
    log_writer.write("Best Model:\n")
    log_writer.flush()
    log_writer.write(s"Lambda: $best_lambda Alpha: $best_alpha Rank: $best_rank Produced RMSE of $best_score\n")
    log_writer.flush()

    val best_model = ALS.trainImplicit(train,best_rank,10,best_lambda,best_alpha)
    val n_test = test.count()
    val best_model_test_rmse = getRMSE(best_model,test,n_test,max_pref,min_pref)
    log_writer.write(s"Best model test dataset RMSE:$best_model_test_rmse")
    log_writer.flush()
    log_writer.close()
  }

}
