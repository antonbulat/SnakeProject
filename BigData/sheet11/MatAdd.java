package bigdata;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import scala.Tuple2;

public class MatAdd {

public static void main(String[] args) {
		SparkConf conf= new SparkConf().setAppName("RelAlg");
		JavaSparkContext sc= new JavaSparkContext(conf);
	
		JavaRDD< List<Long>> matrixA=sc.textFile(args[0])
				.map(l-> {
					List<String> stringNums=Arrays.asList(l.split("\t"));
					List<Long> result = new ArrayList<Long>();
					for (String num : stringNums){
						result.add(Long.parseLong(num));
					}
					return result;});
	
		
		sc.textFile(args[1])
		.map(l-> {
			List<String> stringNums=Arrays.asList(l.split("\t"));
			List<Long> result = new ArrayList<Long>();
			for (String num : stringNums){
				result.add(Long.parseLong(num));
			}
			return result;
			})
			.zip(matrixA)
			.map(x->{
				for(int i=0; i<x._1().size();i++){
					x._1().set(i, x._1().get(i)+x._2().get(i));
				}
				return x;
			})
			.saveAsTextFile(args[2]);
		
		
		sc.close();
		
	}
}
