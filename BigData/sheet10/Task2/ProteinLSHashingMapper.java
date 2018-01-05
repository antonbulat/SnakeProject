package bigdata;


import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Mapper.Context;

public class ProteinLSHashingMapper extends Mapper<Text,Text,Text,Text>{
	
	private Integer bandwidth;
	
	
	
	@Override
	protected void setup(Mapper<Text, Text, Text, Text>.Context context) throws IOException, InterruptedException {
		Configuration conf = context.getConfiguration();
		this.bandwidth = conf.getInt("lsh.bandwidth", 0);
	}

	@Override
	protected void map(Text key, Text value, Context context)
			throws IOException, InterruptedException {
		String[] vals = value.toString().split("\t");
		List<List<Integer>> signs = new ArrayList<List<Integer>>();
		Float numOfBands = ((float)vals.length)/((float)bandwidth);
		int num =(int) Math.ceil(numOfBands);
		
		for(int i = 0 ; i<num;i++){
			List<Integer> band = new ArrayList<Integer>();
			for(int j = 0; j<this.bandwidth&&(j+this.bandwidth*i)<vals.length;j++){
				band.add(Integer.parseInt(vals[j+this.bandwidth*i]));
			}
		}
		
		
	}
	
	private int hashFunktion(String s){
		Integer num = Integer.parseInt(s);
		
	}

}
