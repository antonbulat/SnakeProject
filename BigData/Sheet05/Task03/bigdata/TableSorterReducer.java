package bigdata;


import java.io.IOException;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

public class TableSorterReducer extends Reducer<Text,Text,Text,Text>  {

	@Override
	protected void reduce(Text arg0, Iterable<Text> arg1,Context context)
			throws IOException, InterruptedException {
	
		context.write(arg0, new Text(""));
	}
}
