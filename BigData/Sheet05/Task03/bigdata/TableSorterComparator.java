package bigdata;

import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.WritableComparator;

public class TableSorterComparator extends WritableComparator {
	public TableSorterComparator() {
		super(TableSorterCompositeKey.class, true);
		}
		@Override
		public int compare(WritableComparable w1, WritableComparable w2) {
		TableSorterCompositeKey k1 = (TableSorterCompositeKey)w1;
		TableSorterCompositeKey k2 = (TableSorterCompositeKey)w2;
		
		
		
		if(k1.getSorted().size()==0){// use default value
			return k2.getInput().get(0).compareTo(k1.getInput().get(0));

		}
		for(int i = 0; i<k1.getSorted().size();i++){
			int cmp =  k2.getInput().get(k2.getSorted().get(i))
					.compareTo(k1.getInput().get(k1.getSorted().get(i)));
			if (cmp != 0) {
				return cmp;
			}
		}
		return 0;
		
		}
}
