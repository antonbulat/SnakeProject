package bigdata;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.WritableComparable;

public class TableSorterCompositeKey implements WritableComparable<TableSorterCompositeKey> {
	
	private List<Integer> sorted;
	private List<String> input;

	public void readFields(DataInput in) throws IOException {
	/*
		DoubleWritable temp;
        int l;
        this.input = new ArrayList<String>();
        this.length.readFields(dataInput);
        l = this.length.get();

        for(int i = 0; i < l; i++){
            temp = new DoubleWritable();
            temp.readFields(dataInput);
            this.content.add(temp.get());
        }
        */
	}

	public void write(DataOutput arg0) throws IOException {
		// TODO Auto-generated method stub
		
	}

	
	public List<Integer> getSorted() {
		return sorted;
	}

	public List<String> getInput() {
		return input;
	}

	public int compareTo(TableSorterCompositeKey other) {
		if(sorted.size()==0){// use default value
			return other.input.get(0).compareTo(this.input.get(0));

		}
		for(int i = 0; i<sorted.size();i++){
			int cmp =  other.input.get(sorted.get(i)).compareTo(this.input.get(sorted.get(i)));
			if (cmp != 0) {
			return cmp;
			}
		}
		return 0;
		
	}

	public void set(List<Integer> sorted, List<String> input) {
		this.sorted=sorted;
		this.input=input;
		
	}

}
