package bigdata;

        import org.apache.hadoop.fs.FileSystem;
        import org.apache.hadoop.fs.Path;
        import org.apache.hadoop.mapred.*;
        import org.apache.hadoop.io.Text;

        import java.io.IOException;

public class RelationInputFormat extends FileInputFormat{

    public RelationInputFormat() {
        super();
    }

    protected boolean isSplitable(FileSystem fs, Path filename) {
        return false;
    }

    public RecordReader getRecordReader(InputSplit split, JobConf job, Reporter reporter) throws IOException {
        return (RecordReader) new RelationRecordReader();
    }
}


