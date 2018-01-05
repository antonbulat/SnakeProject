package bigdata;


import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.KeyValueTextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

public class ProteinMinHashing {

	public static void main(String[] args) throws IOException, ClassNotFoundException, InterruptedException {
		Configuration conf = new Configuration();
		GenericOptionsParser options = new GenericOptionsParser(conf, args);
        String remainingArguments[] = options.getRemainingArgs();
        
		 Job job= Job.getInstance(conf, "ProteinMinHashing");
		 
		 job.setJarByClass(ProteinMinHashing.class);
		 job.setMapperClass(ProteinMinHashingMapper.class);
		 job.setReducerClass(ProteinMinHashingReducer.class);
		
		 
		 job.setMapOutputKeyClass(IntWritable.class);
		 job.setMapOutputKeyClass(Text.class);
		 job.setOutputKeyClass(Text.class);
		 job.setOutputValueClass(IntWritable.class);
		 
		 job.addCacheFile(new Path(remainingArguments[1]).toUri());
		 
		 FileInputFormat.addInputPath(job, new Path(remainingArguments[0]));
		 FileOutputFormat.setOutputPath(job, new Path(remainingArguments[2]));
		 System.exit(job.waitForCompletion(true) ? 0 : 1);

	}

}
