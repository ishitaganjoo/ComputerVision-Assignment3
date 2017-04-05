#include <cassert>
#include <fstream>
#include <string>
using namespace std;

class Deep : public Classifier
{
public:
  Deep(const vector<string> &_class_list) : Classifier(_class_list) {}
 
  // Deep training. 
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
    ofstream output("deep_train.dat");
    for(Dataset::const_iterator c_iter=filenames.begin(); c_iter != filenames.end(); ++c_iter)
      {   
          // Get each image
          for(int i=0; i<c_iter->second.size(); i++){
          	 CImg<double> sampled_image = extract_features(c_iter->second[i].c_str());
          	 sampled_image.save_png(("deep_model." + c_iter->first + ".png").c_str());
              // Call overfeat for each image.
              //string name = c_iter->second[i];
          	 string name = "deep_model." + c_iter->first + ".png";
              string str = "./overfeat/bin/linux_64/overfeat -f "+name+" > overfeatTemp";
              const char *command = str.c_str();
              system(command);
              //cout <<"DONE "<< name<<endl;
              int count = 1;
              output << classLabels[c_iter->first] <<" ";// save class
              bool flag = false;
              // Read features
              ifstream file;
              file.open("overfeatTemp");
              string line;
              getline(file, line);              
              while(getline(file, line)) {  
                istringstream splitstring(line);
                    string token;
                    //if (flag)
                      //break;
                     
                     while(getline(splitstring, token, ' ')) {
                     // if (count > 1000){
                       // flag = true;
                       //break;
                      //}
                      output << count << ":" <<token <<" ";
                      count ++;
                      }
                  }
                  output<<endl;
                  file.close();
           }
      }
      

  // Train the svm
  flush(output);
  string str = "./svm_multiclass_learn -c 0.1 -# 1 deep_train.dat deep_model";
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
   CImg<double> sampled_image = extract_features(filename.c_str());
   sampled_image.save_png("deep_model_test.png");
   //string name = filename;
   string name = "deep_model_test.png";
   string str = "./overfeat/bin/linux_64/overfeat -f "+name+" > overfeatTemp";
   const char *command = str.c_str();
   system(command);
   // Create test file for this test image for SVM to run.
      ofstream output("deep_test.dat");
      int count = 1;
      output << actualClass << " ";
      // Read features
              ifstream file;
              file.open("overfeatTemp");
              string line;
              bool flag = false;
              getline(file, line);              
              while(getline(file, line)) {
              //if (flag)
              //break;  
                istringstream splitstring(line);
                    string token;

                    
                     while(getline(splitstring, token, ' ')) {
                      //if (count > 1000){
                     //flag = true;
                      //break;
                    //}
                      output << count << ":" <<token <<" ";
                      count ++;
                      }
                  }
            output << endl;  
            file.close();    
          flush(output);
    // figure prediction for this using svm
    string str1 = "./svm_multiclass_classify deep_test.dat deep_model prediction";
    
    const char *command1 = str1.c_str();
    //cout<<"call command"<<endl;
    system(command1);
    
    cout<<"Tested"<<endl;
    //Read the prediction 
    string line1;
    ifstream file1;
    file1.open("prediction");
    cout <<"ORIGINAL IS "<<actualClass<<endl;
    while(getline(file1, line1)) {
      cout<< "PREDICTION IS "<<line1<<endl;
      int p = atoi(line1.c_str());
      file1.close();
    return classLabels[p];
    }
  }

  virtual void load_model()
  {
    
  }
protected:
	 static const int size=231;  // subsampled image resolution
  // extract features from an image, which in this case just involves resampling and 
  // rearranging into a vector of pixel data.
  CImg<double> extract_features(const string &filename)
    { // Mode of interpolation
      return (CImg<double>(filename.c_str())).resize(size,size,1,1,3);
    }
};
