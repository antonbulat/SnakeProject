package bigdata;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;


public class MatrixMult {

	public static void main(String[] args) throws IOException, ClassNotFoundException, InterruptedException{
				
		Configuration conf = new Configuration();
		//A is an m-by-n matrix, B is an n-by-l matrix
		conf.set("A", args[0]);
		conf.set("m", args[2]);
		conf.set("l", args[3]);

        Job job = Job.getInstance(conf, "MatrixMult");
        
        job.setJarByClass(MatrixMult.class);
        job.setMapperClass(MatrixMultMapper.class);
        job.setReducerClass(MatrixMultReducer.class);        
        job.setInputFormatClass(TextInputFormat.class);
        
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Text.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        
        FileInputFormat.addInputPaths(job, args[0]+ ',' + args[1]);
        FileOutputFormat.setOutputPath(job, new Path(args[4]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
	}

}

