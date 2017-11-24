package bigdata;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

public class NaturalJoinReducer extends Reducer<Text,Text,Text,Text> {

	@Override
	protected void reduce(Text arg0, Iterable<Text> arg1, Context context)
			throws IOException, InterruptedException {
		List<String> rightRelation = new ArrayList<String>(); // second relation S
		List<String> leftRelation = new ArrayList<String>();// first relation R
		
		for(Text line: arg1){
			String[] vals = line.toString().split("§");
			if(vals[0].equals("R")){
				leftRelation.add(vals[1]);
			}else{
				rightRelation.add(vals[1]);
			}
			if(!rightRelation.isEmpty()&&!leftRelation.isEmpty()){// each list should have at least one element for a natural join
				for(String first : leftRelation){
					for(String second : rightRelation){
						context.write(new Text(""), new Text(first+"\t"+second));
					}
				}
			}
		}
		
	}

	

	
}
