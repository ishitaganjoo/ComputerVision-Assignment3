#include <cassert>
#include <fstream>
#include <string>
using namespace std;

class SimpleBaseline : public Classifier
{
public:
  SimpleBaseline(const vector<string> &_class_list) : Classifier(_class_list) {}
 
  // Simple Baseline training. 
  virtual void train(const Dataset &filenames) 
  { 
     map<string, int> classLabels;
  classLabels["bagel"] = 1;
  classLabels["bread"] = 2;
  classLabels["brownie"] = 3;
  classLabels["chickennugget"] = 4;
  classLabels["churro"] = 5;
  classLabels["croissant"] = 6;
  classLabels["frenchfries"] = 7;
  classLabels["hamburger"] = 8;
  classLabels["hotdog"] = 9;
  classLabels["jambalaya"] = 10;
  classLabels["kungpaochicken"] = 11;
  classLabels["lasagna"] = 12;
  classLabels["muffin"] = 13;
  classLabels["paella"] = 14;
  classLabels["pizza"] = 15;
  classLabels["popcorn"] = 16;
  classLabels["pudding"] = 17;
  classLabels["salad"] = 18;
  classLabels["salmon"] = 19;
  classLabels["scone"] = 20;
  classLabels["spaghetti"] = 21;
  classLabels["sushi"] = 22;
  classLabels["taco"] = 23;
  classLabels["tiramisu"] = 24;
  classLabels["waffle"] = 25;
  map<int, vector<CImg<double> > > outputMap; 
    int count = 0;
    for(Dataset::const_iterator c_iter=filenames.begin(); c_iter != filenames.end(); ++c_iter)
      {

	//cout << "Processing " << filenames.size()<< endl;
	CImg<double> class_vectors(size*size*3, filenames.size(), 1);
	vector<CImg<double> > classVectors;
	// convert each image to be a row of this "model" image
	for(int i=0; i<c_iter->second.size(); i++){
	  CImg<double> features = extract_features(c_iter->second[i].c_str());
    //cout <<features[10]<<endl;
    classVectors.push_back(features);
  }
	outputMap[classLabels[c_iter->first]] = classVectors;
	
      }
      // Write the map to an example file to be used by the SVM trainer.
      ofstream output("train.dat");
     

    for(map<int, vector<CImg<double> > >::iterator it = outputMap.begin();it!=outputMap.end();++it){
       //cout <<"outer loop"<< it->first<<endl;
        for(int i=0; i<it->second.size(); i++)
         { 
         // cout <<"inner loop"<< i<<endl;
          output << it->first << " ";
          int lastIndex = it->second[i].size()-1;
          //cout<<"last index "<<it->second[i][lastIndex]<<endl;
          for (int j=0; j<lastIndex; j++){
          //  cout<<"innermost loop" <<j<<endl;
           output<< (j+1) <<":"<< it->second[i][j] << " ";
          }
          output<< (lastIndex+1)<< ":" << it->second[i][lastIndex]<<"\n";
         }

  }
   flush(output);
  string str = "./svm_multiclass_learn -c 0.1 -# 11000 -t 0 train.dat model";
   const char *command = str.c_str();
   system(command);
}

  virtual string classify(const string &filename, int actualClass)
  {  map<int, string> classLabels;
  classLabels[1] = "bagel";
  classLabels[2] = "bread";
  classLabels[3] = "brownie";
  classLabels[4] = "chickennugget";
  classLabels[5] =  "churro";
  classLabels[6] =  "croissant";
  classLabels[7] =  "frenchfries";
  classLabels[8] =  "hamburger";
  classLabels[9] =  "hotdog";
  classLabels[10] =  "jambalaya";
  classLabels[11] =  "kungpaochicken";
  classLabels[12] = "lasagna";
  classLabels[13] = "muffin";
  classLabels[14] = "paella";
  classLabels[15] = "pizza";
  classLabels[16] = "popcorn";
  classLabels[17] = "pudding";
  classLabels[18] = "salad";
  classLabels[19] = "salmon";
  classLabels[20] = "scone";
  classLabels[21] = "spaghetti";
  classLabels[22] = "sushi";
  classLabels[23] = "taco";
  classLabels[24] = "tiramisu";
  classLabels[25] = "waffle";
    CImg<double> test_image = extract_features(filename);
	 // Create test file for this test image for SVM to run.
      ofstream output("test.dat");
      
      output << actualClass << " ";
      int lastIndex = test_image.size()-1;
      for (int j=0; j<lastIndex; j++)
      {
          //  cout<<"innermost loop" <<j<<endl;
           output<< (j+1) <<":"<< test_image[j] << " ";
      }
          output<< (lastIndex+1)<< ":" << test_image[lastIndex]<<"\n";
          flush(output);
    // figure prediction for this using svm
    string str = "./svm_multiclass_classify test.dat model prediction";
    cout<<str<<endl;
    const char *command = str.c_str();
    cout<<"call command"<<endl;
    system(command);
    
    cout<<"Tested"<<endl;
    //Read the prediction 
    string line;
    ifstream file;
    file.open("prediction");
    cout <<"ORIGINAL IS "<<actualClass<<endl;
    while(getline(file, line)) {
    	cout<< "PREDICTION IS "<<line<<endl;
      int p = atoi(line.c_str());
    return classLabels[p];
    }
  }

  virtual void load_model()
  {
    
  }
protected:
  // extract features from an image, which in this case just involves resampling and 
  // rearranging into a vector of pixel data.
  CImg<double> extract_features(const string &filename)
    { // Mode of interpolation
      return (CImg<double>(filename.c_str())).resize(size,size,1,3,3).unroll('x');
    }

  static const int size=25;  // subsampled image resolution
  map<string, CImg<double> > models; // trained models
};
