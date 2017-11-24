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

public class NaturalJoin {

	public static void main(String[] args) throws IOException, ClassNotFoundException, InterruptedException {
		Configuration conf = new Configuration();
		GenericOptionsParser options = new GenericOptionsParser(conf, args);
        String remainingArguments[] = options.getRemainingArgs();
	 
		 Job job= Job.getInstance(conf, "NaturalJoin");
		 
		 job.setJarByClass(NaturalJoin.class);
		 job.setMapperClass(NaturalJoinMapper.class);
		 job.setReducerClass(NaturalJoinReducer.class);
		 job.setInputFormatClass(KeyValueTextInputFormat.class);
		 
		 job.setMapOutputKeyClass(Text.class);
		 job.setMapOutputKeyClass(Text.class);
		 job.setOutputKeyClass(Text.class);
		 job.setOutputValueClass(Text.class);
		 
		 FileInputFormat.addInputPaths(job, remainingArguments[0]+','+remainingArguments[1]);
		 FileOutputFormat.setOutputPath(job, new Path(remainingArguments[2]));
		 System.exit(job.waitForCompletion(true) ? 0 : 1);

	}

}
