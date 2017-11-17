package bigdata;


import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

public class TableSorterReducer extends Reducer<TableSorterCompositeKey,Text,Text,Text>  {

}
