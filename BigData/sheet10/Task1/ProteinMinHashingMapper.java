package bigdata;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.util.StringUtils;

public class ProteinMinHashingMapper extends Mapper<Object,Text,Text,IntWritable>{

	private int shingleNum;
	private int hashNum;
	private List<Integer> aList;
	private List<Integer> bList;
	private final Integer prime = 92821; // Recommended by Hans-Peter Störr from stackoverflow <3
	
	@Override
	protected void map(Object key, Text value,Context context)
			throws IOException, InterruptedException {
		String[] vals=value.toString().split("\t");
		String strI=vals[0];
		String strJ=vals[1];
		Integer i = Integer.parseInt(strI);
		for(int k=0;k<hashNum;k++){
			String newKey = k+"\t"+strJ;
			context.write(new Text(newKey), new IntWritable(executeHash(k,i)));
		}
		
	}

	private Integer executeHash(int k, Integer i) {
		return  ((aList.get(k)*i+bList.get(k))% prime)% shingleNum;
		
	}

	@Override
	protected void setup(Context context) throws IOException, InterruptedException {
		Configuration conf= context.getConfiguration();
		hashNum = conf.getInt("minhash.functions", 10);
		URI[] translationURIs= Job.getInstance(conf).getCacheFiles();
		// we have only one file
		Path translationPath= new Path(translationURIs[0].getPath());
		String translationFileName= translationPath.toString();
		parseTransFile(translationFileName);
		
		//init hash values
		Random rm = new Random(42);
		aList= new ArrayList<Integer>();
		bList = new ArrayList<Integer>();
		for(int k=0;k<hashNum;k++){
			
			aList.add(rm.nextInt());
			bList.add(rm.nextInt() );
		}
		
		
	}
	private void parseTransFile(String translationFileName) {
		shingleNum=0;
		try{
			BufferedReader bf = new BufferedReader(new FileReader(translationFileName));
			String pattern= null;
			String[] st=null;
			String [] trans=null;
			while((pattern= bf.readLine()) != null){
				shingleNum++;
			}
			bf.close();
		}catch(IOException e){
			System.err.println("Caught exception while parsing the cachedfile :("+ StringUtils.stringifyException(e));
		}
		
		
	}
	

}
