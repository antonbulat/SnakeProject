package bigdata;

import java.io.IOException;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

public class ProteinMinHashingReducer extends Reducer<Text,IntWritable,Text,IntWritable>{

	@Override
	protected void reduce(Text arg0, Iterable<IntWritable> arg1, Context arg2)
			throws IOException, InterruptedException {
		
		int min = Integer.MAX_VALUE;
		for(IntWritable val :arg1){
			if(val.get()<min){
				min= val.get();
			}
		}
		arg2.write(arg0, new IntWritable(min));
	}

	
}
