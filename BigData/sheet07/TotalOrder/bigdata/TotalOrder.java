package bigdata;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.KeyValueTextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.partition.InputSampler;
import org.apache.hadoop.mapreduce.lib.partition.TotalOrderPartitioner;

public class TotalOrder {
	public static void main(String[] args) throws IOException, ClassNotFoundException, InterruptedException {
		
		Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "TotalOrder");
        job.setJarByClass(TotalOrder.class);

        job.setNumReduceTasks(10);//Set 10 Reduce Tasks
        FileInputFormat.setInputPaths(job, new Path(args[0]));
  
        /*
        // Create a new file...
        Path partitionPath = new Path(new Path(args[1]) + "-part.lst");
        FileSystem fs = FileSystem.get(conf);
        FSDataOutputStream recOutputWriter = fs.create(partitionPath);
        recOutputWriter.close();
        fs.close();
        //System.setProperty("HADOOP_USER_NAME", "vagrant");
        TotalOrderPartitioner.setPartitionFile(conf, partitionPath);
        */
        job.setInputFormatClass(KeyValueTextInputFormat.class);
        job.setMapOutputKeyClass(Text.class);
        
        //InputSampler.writePartitionFile(job, new InputSampler.SplitSampler<Object, Object>(1000));// for task b
        InputSampler.writePartitionFile(job, new InputSampler.RandomSampler<Object, Object>(0.5,1000));
 
        job.setPartitionerClass(TotalOrderPartitioner.class);
        job.setMapperClass(Mapper.class);
        job.setReducerClass(Reducer.class);
 
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
	}

}
