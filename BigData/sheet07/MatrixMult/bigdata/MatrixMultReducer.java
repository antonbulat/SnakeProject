package bigdata;

import java.io.IOException;
import java.util.HashMap;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

public class MatrixMultReducer extends Reducer<Text, Text, Text, Text>{
	
	@Override
	public void reduce(Text key, Iterable<Text> values, Context context)
			throws IOException, InterruptedException {
		String[] value;
		
		HashMap<Integer, Float> hashA = new HashMap<Integer, Float>();
		HashMap<Integer, Float> hashB = new HashMap<Integer, Float>();
		for (Text v : values) {
			value = v.toString().split("\t");
			if (value[0].equals("A")) {
				hashA.put(Integer.parseInt(value[1]), Float.parseFloat(value[2]));
			} else {
				hashB.put(Integer.parseInt(value[1]), Float.parseFloat(value[2]));
			}
		}
		
//		int n = Integer.parseInt(context.getConfiguration().get("n"));
		int n = 2;  //erstmal hardcoded, wie komme daran?
		float result = 0.0f;
		float a_ij;
		float b_jk;
		Text outputKey = new Text();
		Text outputValue = new Text();
		
		
		for (int j = 0; j< n; j++) {
			outputKey.set(key.toString());
			a_ij = hashA.containsKey(j) ? hashA.get(j) : 0.0f;
			b_jk = hashB.containsKey(j) ? hashB.get(j) : 0.0f;
			result += a_ij * b_jk;
			
		}
		if (result != 0.0f) {
			outputValue.set(Float.toString(result));
			//			context.write(null, new Text(key.toString() + "\t" + Float.toString(result)));
			context.write(outputKey, outputValue);

		}		
	}	
}
