import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashMap;

public class HW3 {

	/**
	 * Runs the tests for HW3
	 */
	public static void main(String[] args) {
		if (args.length < 4) {
			System.out
			.println("usage: java HW3 <modeFlag> <trainFilename> <testFilename> <k instances per-leaf>");
			System.exit(-1);
		}

		/*
		 * mode 0 : output the mutual information of each attribute at the root node 1 : create a
		 * decision tree from a training set, output the tree 2 : create a decision tree from a training
		 * set, output the classifications of a test set 3 : create a decision tree from a training set
		 * then tune, output the tree 4 : create a decision tree from a training set then tune, output
		 * the classifications of a test set
		 */
		int mode = Integer.parseInt(args[0]);
		if (0 > mode || mode >= 4) {
			System.out.println("mode must be between 0 and 3");
			System.exit(-1);
		}

		//Create train set instances and attributes.
		ArrayList<ArrayList<Double>> mTrainDataSet;
		ArrayList<String> mTrainAttributeNames;
		ArrayList<ArrayList<Double>> mTestDataSet;
		ArrayList<String> mTestAttributeNames;
		//Make train data set.
		HashMap<String, Object> mHashMap = createDataSet(args[1]);
		mTrainDataSet = (ArrayList<ArrayList<Double>>)mHashMap.get("mDataSet");
		mTrainAttributeNames = (ArrayList<String>)mHashMap.get("mAttributeNames");
		//Make test data set.
		mHashMap = createDataSet(args[2]);
		mTestDataSet = (ArrayList<ArrayList<Double>>)mHashMap.get("mDataSet");
		mTestAttributeNames = (ArrayList<String>)mHashMap.get("mAttributeNames");

		//Build tree.
		DecisionTreeImpl mTree = new DecisionTreeImpl(mTrainDataSet, mTrainAttributeNames, Integer.parseInt(args[3]));

		if (mode == 0) {
			mTree.rootInfoGain(mTrainDataSet, mTrainAttributeNames);
		}else if(mode == 1){
			mTree.print();
		}else if(mode == 2){

			int guess = 0;
			int guessC = 0;
			int real = 0;
			int correctC = 0;
			
			for (int i = 0; i < mTrainDataSet.size(); i++)
			{       
				//calculate label at leaf
				guess = mTree.classify(mTrainDataSet.get(i));
				real = mTrainDataSet.get(i).get(mTrainDataSet.get(0).size() - 1).intValue();

				if (guess == real)
				{
					correctC++;
				}

				guessC++;

				System.out.println(guess);
			}

			System.out.println((double)((double)correctC)/((double)guessC));


		}else if(mode == 3){

			int guess = 0;
			int real = 0;
			int guessC = 0;
			int correctC = 0;
			//for each instance
			for (int i = 0; i < mTestDataSet.size(); i++)
			{       
				//calculate label at leaf
				guess = mTree.classify(mTestDataSet.get(i));
				real = mTestDataSet.get(i).get(mTestDataSet.get(0).size() - 1).intValue();

				if (guess == real)
				{
					correctC++;
				}

				guessC++;

				//print guess
				System.out.println(guess);
			}

			System.out.println((double)((double)correctC)/((double)guessC));
			
		}else{
			System.out.println("Invalid mode passed as argument.");
		}
	}

	/**
	 * Converts from text file format to a list of lists. Each sub-list represents a row from the file. 
	 * You DO NOT have to use this data format if you don't want to. Use whatever data structure you 
	 * find most convenient.
	 */
	private static HashMap<String, Object> createDataSet(String file) {
		//List of lists. mDataSet.get(i) corresponds to row i from the input file.
		HashMap<String, Object> mHashMap = new HashMap<String, Object>();
		ArrayList<ArrayList<Double>> mDataSet = new ArrayList<ArrayList<Double>>();
		ArrayList<String> mAttributeNames = new ArrayList<String>();  

		BufferedReader in;
		try {
			in = new BufferedReader(new FileReader(file));
			while (in.ready()) {
				String line = in.readLine();
				String prefix = line.substring(0, 2);
				if (prefix.equals("//")) {
				} else if (prefix.equals("##")) {
					mAttributeNames.add(line.substring(2));
				} else {
					String[] splitString = line.split(",");
					//Create data row.
					ArrayList<Double> row = new ArrayList<Double>();
					for(String attr : splitString){
						row.add(Double.parseDouble(attr));
					}
					//Add data row to data table.
					mDataSet.add(row);
				}
			}
			in.close();
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(-1);
		}

		//Add data members to hash map to return.
		mHashMap.put("mDataSet", mDataSet);
		mHashMap.put("mAttributeNames", mAttributeNames);
		return mHashMap;
	}
}