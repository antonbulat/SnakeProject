package bigdata;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

public class MatrixMultMapper extends Mapper<LongWritable, Text, Text, Text> {

	private String r;
	private int m;
	private int l;
	
	@Override
	public void setup(Context context){
		Configuration conf = context.getConfiguration();
		this.r = conf.get("A");
		this.m = Integer.parseInt(conf.get("m"));
		this.l = Integer.parseInt(conf.get("l"));
	}
	
	@Override
	protected void map(LongWritable key, Text value, Context context)
			throws IOException, InterruptedException {
		
//		if(key.toString().endsWith(this.r)){
//			key.set("A");
//		} else {
//			key.set("B");
//		}
		
		
		String[] tripel = value.toString().split("\t");
		Text outputKey = new Text();
		Text outputValue = new Text();
		
		/*
		if(key.toString().endsWith(this.r)){
			for (int k = 0; k < l; k++){
				//outputKey.set(i,k);
				outputKey.set(tripel[0] + "\t" + k);
				//outputValue.set("A", j, a_ij);
				outputValue.set("A" + "\t" + tripel[1] + "\t" + tripel[2]);
				context.write(outputKey, outputValue);
			}
			
		} else {
			for (int k = 0; k < m; k++){
				//outputKey.set(k, j);
				outputKey.set(k + "\t" + tripel[1]);
				//outputValue.set("B", i, b_ij);
				outputValue.set("B" + "\t" + tripel[0] + "\t" + tripel[2]);
				context.write(outputKey, outputValue);
			}
			
		}
		*/
		
		if(tripel[0].equals("A")){
			for (int k = 0; k < l; k++){
				//outputKey.set(i,k);
				outputKey.set(tripel[1] + "\t" + k);
				//outputValue.set("A", j, a_ij);
				outputValue.set("A" + "\t" + tripel[2] + "\t" + tripel[3]);
				context.write(outputKey, outputValue);
			}
			
		} else {
			for (int k = 0; k < m; k++){
				//outputKey.set(k, j);
				outputKey.set(k + "\t" + tripel[2]);
				//outputValue.set("B", i, b_ij);
				outputValue.set("B" + "\t" + tripel[1] + "\t" + tripel[3]);
				context.write(outputKey, outputValue);
			}
			
		}

	}

}
