package bigdata;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;

public class SimpleStatisticsCombiner extends Reducer<Text,DoubleWritable,Text,DoubleWritable> {

    private DoubleWritable resultSum = new DoubleWritable();
    private DoubleWritable resultMax = new DoubleWritable();

    public void reduce(Text key, Iterable<DoubleWritable> values, Context context)
        throws IOException, InterruptedException {
            double temp;
            double max = Double.MIN_VALUE;
            double sum = 0.0;
            for(DoubleWritable d : values){
                temp = d.get();
                if (key.toString().startsWith("Avg")){
                    context.write(key, d);
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

            resultMax.set(max);
            resultSum.set(sum);

            if (key.toString().startsWith("Sum")){
                context.write(key, resultSum);
            }
            else if (key.toString().startsWith("Max")) {
                context.write(key, resultMax);
            }
    }
}
