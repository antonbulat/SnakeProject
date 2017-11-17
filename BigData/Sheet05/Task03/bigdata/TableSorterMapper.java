package bigdata;


import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;


public class TableSorterMapper extends Mapper<Text,Text,TableSorterCompositeKey,Text> {

	
	private List<Integer> sorted;
	
	public void setup(Context context)
		throws IOException, InterruptedException {
		
		Configuration conf = context.getConfiguration();
		int[] sorted_by= conf.getInts("sorter.orderby");
		sorted = new ArrayList<Integer>();
		for(int i =0; i<sorted_by.length;i++){
			sorted.add(sorted_by[i]);
		}
		
	}
	
	@Override
	public void map(Text key, Text value, Context context) throws IOException, InterruptedException {
	
		if(!key.toString().startsWith("#")){
            StringTokenizer st = new StringTokenizer(value.toString());
            List<String> input = new ArrayList<String>();
            input.add(key.toString());
            while(st.hasMoreElements()){
            	input.add(st.nextToken());
            }
         
            TableSorterCompositeKey outputKey = new TableSorterCompositeKey();
            outputKey.set(sorted, input);
    		context.write(outputKey, value);
        }
		
	}
}
