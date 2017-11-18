package bigdata;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.KeyValueTextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;
public class SimpleStatistics{
    public static void main(String[] args) throws Exception{
        Configuration conf = new Configuration();
        
        GenericOptionsParser options = new GenericOptionsParser(conf, args);
        String remainingArguments[] = options.getRemainingArgs();
        //conf.setInt("stats.col", 1);
        
        Job job= Job.getInstance(conf, "SimpleStatistics");
        job.setJarByClass(SimpleStatistics.class);
        job.setMapperClass(SimpleStatisticsMapper.class);
        job.setReducerClass(SimpleStatisticsReducer.class);
        job.setCombinerClass(SimpleStatisticsCombiner.class);
        
        job.setInputFormatClass(KeyValueTextInputFormat.class);
        
        job.setMapOutputKeyClass(IntWritable.class);
        job.setMapOutputValueClass(DoubleWritable.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(DoubleWritable.class);
        
        FileInputFormat.addInputPath(job, new Path(remainingArguments[0]));
        FileOutputFormat.setOutputPath(job, new Path(remainingArguments[1]));
        
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}