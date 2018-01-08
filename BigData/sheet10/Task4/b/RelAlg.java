package bigdata;

import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.JavaRDD;


import java.util.Arrays;


import org.apache.spark.SparkConf;

// exercise (b)
public class RelAlg {

	public static void main(String[] args) {
		
		Integer columnIndex = Integer.parseInt(args[1]);
		String value = args[2];
		
		SparkConf conf= new SparkConf().setAppName("RelAlg");
		JavaSparkContext sc= new JavaSparkContext(conf);
		
		JavaRDD<String> result=sc.textFile(args[0])
				.map(l-> Arrays.asList(l.split("\t")))
				.filter(l->l.get(columnIndex).equals(value))
				.map(l->{
					StringBuilder sb = new StringBuilder();
					for (String s : l){
						sb.append(s).append("\t");
					}
					sb.setLength(sb.length() - 1);
					return sb.toString();
				});
		result.saveAsTextFile(args[3]);	
		sc.close();
	}
}
