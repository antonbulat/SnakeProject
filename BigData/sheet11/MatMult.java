package bigdata;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import scala.Tuple2;

public class MatMult {

public static void main(String[] args) {
		
		SparkConf conf= new SparkConf().setAppName("RelAlg");
		JavaSparkContext sc= new JavaSparkContext(conf);
		
		JavaPairRDD<Integer, List<Integer>> matrixA=sc.textFile(args[0])
				.mapToPair(l-> {
					List<String> stringNums=Arrays.asList(l.split("\t"));
					List<Integer> result = new ArrayList<Integer>();
					result.add(0);// 0 stands for matrix A
					result.add(Integer.parseInt(stringNums.get(0)));//i
					result.add(Integer.parseInt(stringNums.get(2)));//aij
					return new Tuple2<Integer,List<Integer>>(Integer.parseInt(stringNums.get(1)),result);});
	
		
		sc.textFile(args[1])
			.mapToPair(l-> {
				List<String> stringNums=Arrays.asList(l.split("\t"));
				List<Integer> result = new ArrayList<Integer>();
				result.add(1);// 1 stands for matrix B
				result.add(Integer.parseInt(stringNums.get(0)));//i
				result.add(Integer.parseInt(stringNums.get(2)));//bij
				return new Tuple2<Integer,List<Integer>>(Integer.parseInt(stringNums.get(1)),result);
				})
			.union(matrixA)
			.groupByKey()
			.flatMap(t->{
				List<Tuple2<Integer,Integer>> lA= new ArrayList<Tuple2<Integer,Integer>>();
				List<Tuple2<Integer,Integer>> lB= new ArrayList<Tuple2<Integer,Integer>>();
				for(List<Integer> list: t._2()){
					if(list.get(0)==0){
						lA.add(new Tuple2<Integer,Integer>(list.get(1),list.get(2)));
					}else{
						lB.add(new Tuple2<Integer,Integer>(list.get(1),list.get(2)));
					}
				}
				List<Tuple2<Tuple2<Integer,Integer>,Integer>> result= new ArrayList<>();
				for(Tuple2<Integer,Integer> a: lA){
					for(Tuple2<Integer,Integer> b: lB){
						result.add(new Tuple2<Tuple2<Integer,Integer>,Integer>(new Tuple2<Integer,Integer>(a._1(),b._1()),a._2()*b._2()));
					}
				}
				return result;
			})
			.mapToPair(x->x)
			.reduceByKey((x,y)->x+y)
			.saveAsTextFile(args[2]);
		
		
		sc.close();
		
	}
}
