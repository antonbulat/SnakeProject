package bigdata;


import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.KeyValueTextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

public class ProteinLSHashing {

	public static void main(String[] args) throws IOException, ClassNotFoundException, InterruptedException {
		Configuration conf = new Configuration();
		GenericOptionsParser options = new GenericOptionsParser(conf, args);
        String rem[] = options.getRemainingArgs();
        
		 Job job= Job.getInstance(conf, "ProteinLSHashing");
		 
		 job.setJarByClass(ProteinLSHashing.class);
		 job.setMapperClass(ProteinLSHashingMapper.class);
		 job.setReducerClass(ProteinLSHashingReducer.class);
		 
		 job.setInputFormatClass(KeyValueTextInputFormat.class);
		 
		 
		 FileInputFormat.addInputPath(job, new Path(rem[0]));
		 FileOutputFormat.setOutputPath(job, new Path(rem[1]));
		 System.exit(job.waitForCompletion(true) ? 0 : 1);

	}
}
