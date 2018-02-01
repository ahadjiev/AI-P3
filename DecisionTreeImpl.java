import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

import org.w3c.dom.Node;

/**
 * Fill in the implementation details of the class DecisionTree using this file. Any methods or
 * secondary classes that you want are fine but we will only interact with those methods in the
 * DecisionTree framework.
 * 
 * You must add code for the 1 member and 4 methods specified below.
 * 
 * See DecisionTree for a description of default methods.
 */
public class DecisionTreeImpl{
	private DecTreeNode root;


	//ordered list of attributes
	//NAMES
	private List<String> mTrainAttributes; 

	String bestSplitPointList [][];
	//

	//DaTA	last column is t/f
	private ArrayList<ArrayList<Double>> mTrainDataSet;
	//Min number of instances per leaf.
	private int minLeafNumber = 10;

	/**
	 * Answers static questions about decision trees.
	 */
	DecisionTreeImpl() {
		// no code necessary this is void purposefully
	}

	/**
	 * Build a decision tree given a training set then prune it using a tuning set.
	 * 
	 * @param train: the training set
	 * @param tune: the tuning set
	 */
	DecisionTreeImpl(ArrayList<ArrayList<Double>> trainDataSet,
			ArrayList<String> trainAttributeNames, int minLeafNumber) {
		this.mTrainAttributes = trainAttributeNames;
		this.mTrainDataSet = trainDataSet;
		this.minLeafNumber = minLeafNumber;
		this.root = buildTree(this.mTrainDataSet);
	}

	private DecTreeNode buildTree(ArrayList<ArrayList<Double>> dataSet){

		if (dataSet.size() <= minLeafNumber)
		{
			//initialize count variables for 1 and 0
			int oneC = 0;
			int zeroC = 0;
			//increment oneC or zeroC depending on value of current label
			for (int i = 0; i < dataSet.size(); i++)
			{
				if(dataSet.get(i).get(dataSet.get(0).size() - 1).equals(0.0))
				{
					zeroC++;
				}else{
					oneC++;
				}
			}
			
			//if majority is 0 then return the leaf without label 0
			DecTreeNode node;
			if (zeroC > oneC)
			{
				node = new DecTreeNode(0,"hi", 10000);
			}else{
				//if 1 is major thao or equal, return leaf w 1 label
				node = new DecTreeNode(1,"hi", 10000);
			}
			return node;
		}
		
		//initialize count variables for 1 and 0
		int zeroC = 0;
		int oneC = 0;
		//for each instance
		for (int i = 0; i < dataSet.size(); i++)
		{
			//of label is 0
			if(dataSet.get(i).get(dataSet.get(0).size() - 1).equals(0.0))
			{
				zeroC++;
			}else{
				oneC++;
			}
		}

		//returns a new node using 1 as label if all labels are 1
		if (zeroC == 0)
		{
			return new DecTreeNode(1,"hi", 10000);
		}
		//returns a new node using 0 as label if all labels are 0
		if (oneC == 0)
		{
			return new DecTreeNode(0,"hi", 10000);
		}


		
		DecTreeNode returnTree;
		int temp = 0;
		ArrayList<Double> gainOfAttributes = new ArrayList<Double>();
		ArrayList<ArrayList<Double>> tSort = new ArrayList<ArrayList<Double>>();

		for (int i = 0; i < dataSet.get(0).size() - 1; i++)
		{
			//sort the attribute values
			tSort.clear();
			tSort = sortDoubleDoubleArray(dataSet, i);

			//update array of info gain values
			gainOfAttributes.add(bestGain(tSort, i));

		}
		//calculate max gain
		Double maxGain = 0.0;
		for (int i = 0; i < gainOfAttributes.size() - 1; i++)
		{
			if (gainOfAttributes.get(i) >= maxGain)
			{
				maxGain = gainOfAttributes.get(i);
			}
		}
		//find attribute associated with max gain
		for (int i = 0; i < gainOfAttributes.size() - 1; i++)
		{
			if (gainOfAttributes.get(i).equals(maxGain))
			{
				temp = i;
			}
		}

		//get attribute name
		String attribute = "hi";
		for (int i = 0; i < this.mTrainAttributes.size(); i++)
		{
			if (i == temp)
			{
				attribute = this.mTrainAttributes.get(i);
			}
		}
		//split and sort 
		double splitThresh = bestGainOfThreshold(dataSet, temp);
		dataSet = sortDoubleDoubleArray(dataSet, temp);

		//initialize split ArrayLists 
		ArrayList<ArrayList<Double>> split1 = new ArrayList<ArrayList<Double>>();
		ArrayList<ArrayList<Double>> split2 = new ArrayList<ArrayList<Double>>();
		split1.clear();
		split2.clear();
		for (int k = 0; k < dataSet.size(); k++)
		{		
			//add instance to split 1 if the attribute value is <= threshold
			if (splitThresh >= dataSet.get(k).get(temp))
			{
				split1.add(dataSet.get(k));
			}else{
				//else add instance to split2
				split2.add(dataSet.get(k));
			}
		}

		split1 = sortDoubleDoubleArray(split1, temp);
		split2 = sortDoubleDoubleArray(split2, temp);

		//return binary tree
		returnTree = new DecTreeNode(7,attribute, splitThresh);
		returnTree.left = buildTree(split1);
		returnTree.right = buildTree(split2);
		return returnTree;

	}
	//calculate the log2
	public double log2(double x)
	{
		return ((double)(Math.log(x) / Math.log(2)));
	}
	
	public double calculateEntropy(ArrayList<ArrayList<Double>> dataSet)
	{		
		if (dataSet.isEmpty())
		{
			return 0;
		}
		double eValue = 0;

		double oneC = 0;
		double zeroC = 0;
		//count how many labels are 0 and 1 
		for (int i = 0; i < dataSet.size(); i++)
		{
			//get label, update count for each instance
			if (dataSet.get(i).get(dataSet.get(0).size() - 1) == 0)
			{
				zeroC++;
			}else{
				oneC++;
			}
		}
		//calculate probability of 0 and 1 labels
		double p0 = (double)(zeroC / (zeroC + oneC));
		double p1 = (double)(oneC / (oneC + zeroC));
		if (p1 == 0 || p0 == 0)
		{
			return 0;
		}
		//calculate entropy value and return
		eValue = -(p0*(log2(p0)) + (p1*(log2(p1))));
		return eValue;
	}


	public double bestGainOfThreshold(ArrayList<ArrayList<Double>> dataSet, int attribute)
	{

		dataSet = sortDoubleDoubleArray(dataSet, attribute);


		//initialize threshold candidates
		double bThreshold = 0.0;
		double bInfoGain = 0.0;

		double entropy = calculateEntropy(dataSet);
		double entropy1;
		double entropy2;

		//initialize ArrayList to store threshold values
		ArrayList<Double> thresholds = new ArrayList<Double>();
		thresholds.clear();
		for (int j = 1; j < dataSet.size(); j++)
		{
			//if neighboring labels are different then add to thresholds
			if (!((dataSet.get(j - 1).get(dataSet.get(0).size() - 1)).equals(dataSet.get(j).get(dataSet.get(0).size() - 1))))
			{	
				thresholds.add((double)((double)((double)dataSet.get(j - 1).get(attribute) + (double)dataSet.get(j).get(attribute))/2.0));
			}
		}

		if (thresholds.isEmpty())
		{
			return 0.0;
		}

		//initialize split ArrayLists 
		ArrayList<ArrayList<Double>> split1 = new ArrayList<ArrayList<Double>>();
		ArrayList<ArrayList<Double>> split2 = new ArrayList<ArrayList<Double>>();
		split1.clear();
		split2.clear();	

		for (int j = 0; j < thresholds.size(); j++)
		{
			split1.clear();
			split2.clear();
			for (int k = 0; k < dataSet.size(); k++)
			{			
				//add instance to split 1 if the attribute value is <= threshold
				if (thresholds.get(j) >= dataSet.get(k).get(attribute))
				{
					split1.add(dataSet.get(k));
				}else{
					//else add to split2
					split2.add(dataSet.get(k));
				}
			}
			
			//calculate entropy values of dataSet split1 split2
			entropy = calculateEntropy(dataSet);
			entropy1 = calculateEntropy(split1);
			entropy2 = calculateEntropy(split2);

			//calculate info gain for this threshold candidate
			Double ig = (double)(entropy - (double)((double)split1.size()/(double)dataSet.size())*(entropy1)
					- (double)((double)split2.size()/(double)dataSet.size())*(entropy2));

			if (ig >= bInfoGain)
			{
				bInfoGain = ig;
				bThreshold = thresholds.get(j);
			}
		}


		return bThreshold;
	}

	//best into gain value for attribute
	public double bestGain(ArrayList<ArrayList<Double>> dataSet, int attribute)
	{
		dataSet = sortDoubleDoubleArray(dataSet, attribute);

		//get thresh candidates
		double bestThresh = 0.0;
		double bestIG = 0.0;

		double entropy = calculateEntropy(dataSet);
		double entropy1;
		double entropy2;

		//initialize ArrayList to store threshold values
		ArrayList<Double> thresholds = new ArrayList<Double>();
		thresholds.clear();
		for (int j = 1; j < dataSet.size(); j++)
		{
			//if neighboring labels are different then add to thresholds
			if (!((dataSet.get(j - 1).get(dataSet.get(0).size() - 1)).equals(dataSet.get(j).get(dataSet.get(0).size() - 1))))
			{	
				thresholds.add((double)((double)(dataSet.get(j - 1).get(attribute) + dataSet.get(j).get(attribute))/2));
			}
		}

		if (thresholds.isEmpty())
		{
			return 0.0;
		}

		//initialize split ArrayLists 
		ArrayList<ArrayList<Double>> split1 = new ArrayList<ArrayList<Double>>();
		ArrayList<ArrayList<Double>> split2 = new ArrayList<ArrayList<Double>>();
		split1.clear();
		split2.clear();	

		for (int j = 0; j < thresholds.size(); j++)
		{
			split1.clear();
			split2.clear();
			for (int k = 0; k < dataSet.size(); k++)
			{				

				//add instance to split 1 if the attribute value is <= threshold
				if (thresholds.get(j) >= dataSet.get(k).get(attribute))
				{
					split1.add(dataSet.get(k));
				}else{
					//else add to split2
					split2.add(dataSet.get(k));
				}
			}
			
			entropy = calculateEntropy(dataSet);
			entropy1 = calculateEntropy(split1);
			entropy2 = calculateEntropy(split2);

			//calculate info gain for this threshold candidate
			Double ig = (double)(entropy - (double)((double)split1.size()/(double)dataSet.size())*(entropy1)
					- (double)((double)split2.size()/(double)dataSet.size())*(entropy2));

			if (ig >= bestIG)
			{
				bestIG = ig;
				bestThresh = thresholds.get(j);
			}
		}

		return bestIG;
	}

	public ArrayList<ArrayList<Double>> sortDoubleDoubleArray(ArrayList<ArrayList<Double>> dataSet, int columnToSortBy)
	{
		ArrayList<ArrayList<Double>> temp = new ArrayList<ArrayList<Double>>();
		temp.clear();
		boolean sComplete = false;
		double min = 10000000000.0;
		int minInstance = 0;
		int a = 1;
		//will hold originally gotten instances
		ArrayList<Integer> orig = new ArrayList<Integer>();


		while (temp.size() != dataSet.size())
		{
			//finds min instances not yet copied
			min = 10000000000.0;
			for (int k = 0; k < dataSet.size(); k++)
			{	
				//if instance is smaller than min and not yet copied, new min
				if (dataSet.get(k).get(columnToSortBy) < min)
				{
					if (!orig.contains(k))
					{
						min = dataSet.get(k).get(columnToSortBy);
						minInstance = k;
					}
				}
			}
			temp.add(dataSet.get(minInstance));				
			orig.add(minInstance);
		}

		//sorted table based on label
		ArrayList<Double> hSort = new ArrayList<Double>();
		while (!sComplete)
		{
			sComplete = true;

			for (int k = 0; k < temp.size() - 1; k++)
			{

				if (temp.get(k).get(columnToSortBy).equals(temp.get(k + 1).get(columnToSortBy)))
				{
					if (temp.get(k).get(dataSet.get(0).size() - 1) > temp.get(k + 1).get(dataSet.get(0).size() - 1))
					{
						sComplete = false;
						hSort = temp.get(k);
						temp.set(k, temp.get(k+1));
						temp.set(k + 1, hSort);
					}
				}
			}
		}
		return temp;
	}


	public int classify(List<Double> instance) {
		DecTreeNode curr = root;
		while (!(curr.left == null || curr.right == null))
		{
			int i = Integer.parseInt(curr.attribute.substring(1));
			if ((double)instance.get(i - 1).doubleValue() <= curr.threshold)
			{
				curr = curr.left;
			}else{
				curr = curr.right;
			}
		}
		int classLabel = curr.classLabel;
		return classLabel;
	}

	public void rootInfoGain(ArrayList<ArrayList<Double>> dataSet, ArrayList<String> trainAttributeNames) {
		this.mTrainAttributes = trainAttributeNames;
		for(int i = 0; i < trainAttributeNames.size(); i++){
			System.out.println(this.mTrainAttributes.get(i) + " " + String.format("%.6f", bestGain(dataSet, i)));
		}	
	}


	/**
	 * Print the decision tree in the specified format
	 */
	public void print() {
		printTreeNode("", this.root);
	}

	/**
	 * Recursively prints the tree structure, left subtree first, then right subtree.
	 */
	public void printTreeNode(String prefixStr, DecTreeNode node) {
		String printStr = prefixStr + node.attribute;

		System.out.print(printStr + " <= " + String.format("%.6f", node.threshold));
		if(node.left.isLeaf()){
			System.out.println(": " + String.valueOf(node.left.classLabel));
		}else{
			System.out.println();
			printTreeNode(prefixStr + "|\t", node.left);
		}
		System.out.print(printStr + " > " + String.format("%.6f", node.threshold));
		if(node.right.isLeaf()){
			System.out.println(": " + String.valueOf(node.right.classLabel));
		}else{
			System.out.println();
			printTreeNode(prefixStr + "|\t", node.right);
		}


	}

	public double printAccuracy(int numEqual, int numTotal){
		double accuracy = numEqual/(double)numTotal;
		System.out.println(accuracy);
		return accuracy;
	}

	/**
	 * Private class to facilitate instance sorting by argument position since java doesn't like passing variables to comparators through
	 * nested variable scopes.
	 * */
	private class DataBinder{

		public ArrayList<Double> mData;
		public int i;
		public DataBinder(int i, ArrayList<Double> mData){
			this.mData = mData;
			this.i = i;
		}
		public double getArgItem(){
			return mData.get(i);
		}
		public ArrayList<Double> getData(){
			return mData;
		}

	}

}
