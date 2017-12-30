package bigdata;
import org.apache.spark.api.java.JavaSparkContext;

import scala.Tuple2;

import org.apache.spark.api.java.JavaPairRDD;

import java.util.Arrays;

import org.apache.spark.SparkConf;

public class RelAlg {

	public static void main(String[] args) {
		
		Integer columnIndex = Integer.parseInt(args[1]);
		
		SparkConf conf= new SparkConf().setAppName("RelAlg");
		JavaSparkContext sc= new JavaSparkContext(conf);
		
		JavaPairRDD<String, Integer> result=sc.textFile(args[0])
				.map(l-> Arrays.asList(l.split("\t")).get(columnIndex))
				.mapToPair(w->new Tuple2<String, Integer>(w, 1))
				.reduceByKey((x,y)->x+y);result.saveAsTextFile(args[2]);
				
		sc.close();
	}
}
