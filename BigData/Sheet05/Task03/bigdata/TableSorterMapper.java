package bigdata;


import java.io.IOException;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;


public class TableSorterMapper extends Mapper<Text,Text,Text,Text> {

	@Override
	public void map(Text key, Text value, Context context) throws IOException, InterruptedException {
	
		if(!key.toString().startsWith("#")){
            StringBuilder sb = new StringBuilder(key.toString());
            sb.append("\t").append(value.toString());
            Text t = new Text(sb.toString());
    		context.write(t,t);
        }
		
	}
}
