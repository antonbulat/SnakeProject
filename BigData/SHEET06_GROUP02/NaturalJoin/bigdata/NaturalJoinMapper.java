package bigdata;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

public class NaturalJoinMapper extends Mapper<Text,Text,Text,Text>{
	
	private int leftIndex;
	private int rightIndex;
	
	@Override
	protected void setup(Context context) throws IOException, InterruptedException {
		Configuration conf = context.getConfiguration();
		this.leftIndex = conf.getInt("joincol.left", 0);
		this.rightIndex = conf.getInt("joincol.right", 0);
	}

	@Override
	protected void map(Text key, Text value, Context context)
			throws IOException, InterruptedException {
		
		if(key.toString().equals("R")){
			String[]vals = value.toString().split("\t");
			Text newKey= new Text(vals[this.leftIndex]);
			Text newVal= new Text("R§"+value.toString());// for output order
			context.write(newKey, newVal);
		}else if(key.toString().equals("S")){
			String[]vals = value.toString().split("\t");
			Text newKey= new Text(vals[this.rightIndex]);
			
			String[] valString = value.toString().split("\t");
			StringBuilder sb = new StringBuilder("");// for output order
			for(int i = 0 ;i<valString.length-1;i++){
				if(!(i==this.rightIndex))// delete join attribute in second relation
					sb.append(valString[i].trim()).append("\t");
			}
			sb.append(valString[valString.length-1]);
			Text newVal= new Text("S§"+sb.toString());
			context.write(newKey, newVal);
		}
	}

	
	

}
