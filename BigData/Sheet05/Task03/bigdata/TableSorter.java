package bigdata;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.KeyValueTextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

public class TableSorter {

	public static void main(String[] args) throws Exception{
        Configuration conf = new Configuration();
        
        GenericOptionsParser options = new GenericOptionsParser(conf, args);
        String remainingArguments[] = options.getRemainingArgs();
        
        Job job= Job.getInstance(conf, "TableSorter");
        job.setJarByClass(TableSorter.class);
        job.setMapperClass(TableSorterMapper.class);
        job.setReducerClass(TableSorterReducer.class);
        
        job.setSortComparatorClass(TableSorterComparator.class);// set comparator
        
        
        job.setInputFormatClass(KeyValueTextInputFormat.class);
   
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Text.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        
        FileInputFormat.addInputPath(job, new Path(remainingArguments[0]));
        FileOutputFormat.setOutputPath(job, new Path(remainingArguments[1]));
        
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
