package bigdata;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.IOException;
import java.util.StringTokenizer;

public class SimpleStatisticsMapper extends Mapper<Text,Text,Text,DoubleWritable>{
    
	private int param;
	
	public void setup(Context context)
		throws IOException, InterruptedException {
		
		Configuration conf = context.getConfiguration();
		param = conf.getInt("stats.col", 1);
	}
	
	
	public void map(Text key, Text value, Context context) throws IOException, InterruptedException {
        if(!key.toString().startsWith("#")){
            StringTokenizer st = new StringTokenizer(value.toString());
            int j = 1;
            
            while(st.hasMoreElements()){
            	if(param==j){
					context.write(new Text("Sum_" + param), new DoubleWritable(Double.valueOf(st.nextToken())));
					context.write(new Text("Max_" + param), new DoubleWritable(Double.valueOf(st.nextToken())));
					context.write(new Text("Avg_" + param), new DoubleWritable(Double.valueOf(st.nextToken())));
					break;
				}
            	j++;
            }
            
        }
    }
}