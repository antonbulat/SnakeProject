package bigdata;

import org.apache.hadoop.io.BinaryComparable;
import org.apache.hadoop.io.WritableComparator;
import org.apache.hadoop.mapreduce.RecordReader;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.input.LineRecordReader;

import java.util.ArrayList;
import java.util.List;

import java.io.IOException;
import java.io.InputStream;

public class RelationRecordReader extends RecordReader<Text, TupleWritable> {
        private LineRecordReader lineRecordReader = null;
        private Text key = null;
        private TupleWritable value = null;

        public RelationRecordReader(){

        }

        public void close() throws IOException {
            if (null != lineRecordReader) {
                lineRecordReader.close();
                lineRecordReader = null;
            }
            key = null;
            value = null;
        }

        public Text getCurrentKey() throws IOException, InterruptedException {
            return key;
        }

        public TupleWritable getCurrentValue() throws IOException, InterruptedException {
            return value;
        }

        public float getProgress() throws IOException, InterruptedException {
            return lineRecordReader.getProgress();
        }

        public void initialize(InputSplit split, TaskAttemptContext context) throws IOException, InterruptedException {
            close();

            lineRecordReader = new LineRecordReader();
            lineRecordReader.initialize(split, context);
        }

        public boolean nextKeyValue() throws IOException, InterruptedException {
            if (!lineRecordReader.nextKeyValue()) {
                key = null;
                value = null;
                return false;
            }

            Text line = lineRecordReader.getCurrentValue();
            String str = line.toString();
            String[] arr = str.split("\\t");

            List<String> attribute = new ArrayList<String>();

            for (int i = 0; i < arr.length; i++) {
                attribute.add(arr[i]);
            }
            if (!str.startsWith("#")){
                key = new Text(arr[1]);
                value = new TupleWritable(attribute);
            }
            else{
                key = null;
                value = null;
            }


            return true;
        }

    }

