package bigdata;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;


public class SimpleStatisticsReducer extends Reducer<Text,DoubleWritable,Text,DoubleWritable> {
    public void reduce(Text key, Iterable<DoubleWritable> values, Context context) throws IOException, InterruptedException{
        double temp;
        double max = Double.MIN_VALUE;
        double sum = 0.0;
        int count = 0;
        for(DoubleWritable d : values){
            temp = d.get();
            if (key.toString().startsWith("Avg")){
                sum += temp;
                count++;
            }
            if (key.toString().startsWith("Sum")){
                sum += temp;
            }
            if (key.toString().startsWith("Max")){
                if(temp > max){
                    max = temp;
                }
            }
        }
        double avg = sum/count;

        if (key.toString().startsWith("Avg")){
            context.write(key, new DoubleWritable(avg));
        }
        else if (key.toString().startsWith("Sum")){
            context.write(key, new DoubleWritable(sum));
        }
        else if (key.toString().startsWith("Max")){
            context.write(key, new DoubleWritable(max));
        }

    }
}
