package bigdata;

import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.JavaRDD;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.spark.SparkConf;

//exercise (c)
public class RelAlg {

	public static void main(String[] args) {
		
		List<Integer> indices = new ArrayList<Integer>();
		
		for(String num : args[1].split(",")){
			Integer columnIndex = Integer.parseInt(num);
			indices.add(columnIndex);
		}
		
		SparkConf conf= new SparkConf().setAppName("RelAlg");
		JavaSparkContext sc= new JavaSparkContext(conf);
		
		JavaRDD<String> result=sc.textFile(args[0])
				.map(l->{
					List<String>myList=Arrays.asList(l.split("\t"));
					StringBuilder sb = new StringBuilder();
					for (Integer ind : indices){
						sb.append(myList.get(ind)).append("\t");
					}
					sb.setLength(sb.length() - 1);
					return sb.toString();
				})
				.distinct();
		
		result.saveAsTextFile(args[2]);	
		sc.close();
	}
}
