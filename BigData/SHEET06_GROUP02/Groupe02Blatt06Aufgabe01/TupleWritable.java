package bigdata;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.WritableComparator;

import java.util.List;

public class TupleWritable extends WritableComparator {

    List<String> attribute;

    protected TupleWritable(){
        super(Text.class, true);
    }

    public TupleWritable (List<String> atr){
        this.attribute = atr;
    }

    public boolean equals(List<String> obj) {
        boolean isEqual = true;
        if (this.attribute.size() != obj.size()){
            return false;
        }
        for (int i = 0; i < this.attribute.size(); i++){
            if (!attribute.get(i).equals(obj.get(i))){
                isEqual = false;
            }
        }
        return isEqual;
    }

    public int hashCode() {
        int hc = 17;
        int hashMultiplier = 59;
        hc = hc * hashMultiplier + this.attribute.size();
        for (int i=0; i < this.attribute.size(); i++) {
            hc = hc * hashMultiplier + this.attribute.get(i).hashCode();
        }
        return hc;
    }

    public int compare(List<String> a, List<String> b) {
        if (!a.equals(b)){
            for (int i = 0; i < a.size(); i++){
                if (!a.get(i).equals(b.get(i))) {
                    for (int j = 0; j < a.size(); j++) {
                        if (a.get(i).compareTo(b.get(i)) > 0)
                            return 1;
                        else
                            return -1;
                    }
                }
            }
        }
        return 0;
    }
}
