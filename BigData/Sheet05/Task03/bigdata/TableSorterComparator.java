package bigdata;

import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.WritableComparator;
import org.apache.hadoop.io.Text;

public class TableSorterComparator extends WritableComparator {
	
	private List<Integer> sorted;
	
	
	public TableSorterComparator() {
		super(Text.class,true);
		}
		@Override
		public int compare(WritableComparable w1, WritableComparable w2) {
			Text k1 = (Text)w1;
			Text k2 = (Text)w2;
			
			String[] t2=k1.toString().split("\t");
			String[] t1=k2.toString().split("\t");
			
			if(sorted.size()==0){// use default value
				return t2[0].compareTo(t1[0]);
		
			}
			for(int i = 0; i<sorted.size();i++){
				int cmp = t2[i].compareTo(t1[i]);
				if (cmp != 0) {
					return cmp;
				}
			}
			return 0;
			
		
		}
		@Override
		public void setConf(Configuration conf) {
			// TODO Auto-generated method stub
			int[] sorted_by= conf.getInts("sorter.orderby");
			sorted = new ArrayList<Integer>();
			for(int i =0; i<sorted_by.length;i++){
				sorted.add(sorted_by[i]);
			}
			super.setConf(conf);
		}
}
