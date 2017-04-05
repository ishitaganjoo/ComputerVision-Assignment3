#include "CImg.h"
#include <stdlib.h> 
#include <sstream>
class Bow : public Classifier
{
public:
  Bow(const vector<string> &_class_list) : Classifier(_class_list) {}
  
  // Make Bow model
  virtual void train(const Dataset &filenames) 
  { 
         // Create a bag of all descriptors
    //  vector<vector<SiftDescriptor> > bagOfDescriptors;
    //  ofstream output("descriptors.csv");
    // for(Dataset::const_iterator c_iter=filenames.begin(); c_iter != filenames.end(); ++c_iter)
    //   { cout<< "Loading "<< c_iter->first<<endl;
    //     for(int i=0; i<c_iter->second.size(); i++){
    //     // Compute SIFT for each image and create bag of sift vectors.
    //     CImg<double> input_image(c_iter->second[i].c_str());
    //     CImg<double> gray = (input_image).get_RGBtoHSI().get_channel(2);
    //     vector<SiftDescriptor> descriptors = Sift::compute_sift(gray);
    //     //cout<< descriptors[1].descriptor[1]<<"hello"<<endl;
    //     for (int i = 0; i< descriptors.size(); i++){
    //      for (int j = 0; j< 127; j++) 
    //        {
    //       output << descriptors[i].descriptor[j] << ",";
    //       }
    //     output << descriptors[i].descriptor[127]<<endl;
    //     }
    //   }

  //}
        // The entire k means is taken from mlpack http://www.mlpack.org/docs/mlpack-2.2.0
        //string str = "./mlpack_kmeans -c 25 -i descriptors.csv -C centroids.csv";
       // const char *command = str.c_str();
        //system(command); 
        createHistogram(filenames);
  }

  virtual string classify(const string &filename, int actualClass = 0)
  { 
     map<int, string> classLabels;
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
   
   // Create test file for this test image for SVM to run.
      ofstream output("test_bow.dat");
      output << actualClass<<" ";
      CImg<double> input_image(filename.c_str());
      CImg<double> gray = (input_image).get_RGBtoHSI().get_channel(2);
      vector<SiftDescriptor> descriptors = Sift::compute_sift(gray);
     // Read the centoids.csv and create visual vocabulary
    // http://www.cplusplus.com/forum/general/13087/
       vector<vector<float> >visualVocabulary;
      ifstream filetest ( "centroids.csv" ); 
       string value;
       while(getline ( filetest, value)) 
        {         
          vector<float> visualWord;
          // http://stackoverflow.com/questions/11719538/how-to-use-stringstream-to-separate-comma-separated-strings
          istringstream splitstring(value);
          string token;
          while(getline(splitstring, token, ',')) {
             float parseToken = atof(token.c_str());
             //cout<<parseToken<<endl;
             visualWord.push_back(parseToken);
              }
              visualVocabulary.push_back(visualWord);
         } 
         //cout<< "first "<< visualVocabulary[0][5];
    
        int count = visualVocabulary.size();
        int histogram;
        for(int j = 0; j < count-1; j++)
            { histogram = 0;
              for (int i = 0; i < descriptors.size(); i ++)
              {            
              histogram += euclideanDistance(descriptors[i].descriptor, visualVocabulary[j]);             
            } 
              if (histogram != 0)
              output << (j+1) << ":" << histogram << " ";
          }
          // Last visual word.

              histogram = 0;
              for (int i = 0; i < descriptors.size(); i ++)
              { 
              histogram += euclideanDistance(descriptors[i].descriptor, visualVocabulary[count-1]);
            } 
              if (histogram != 0)
              output << count << ":" << histogram;
            output << endl;
          
          flush(output);
    // figure prediction for this using svm
    string str = "./svm_multiclass_classify test_bow.dat model_bow prediction";
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
      return (CImg<double>(filename.c_str())).resize(size,size,1,1,3);
    }
  static const int size=0;  // subsampled image resolution
  map<string, CImg<double> > models; // trained models

  // Create histogram of each training image
   void createHistogram(const Dataset &filenames)
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
    vector<vector<float> >visualVocabulary;
    // Read the centoids.csv and create visual vocabulary
    // http://www.cplusplus.com/forum/general/13087/
      ifstream file ( "centroids.csv" ); 
        string value;
       while(getline ( file, value)) 
        {         
          vector<float> visualWord;
          // http://stackoverflow.com/questions/11719538/how-to-use-stringstream-to-separate-comma-separated-strings
          istringstream splitstring(value);
          string token;
          while(getline(splitstring, token, ',')) {
             float parseToken = atof(token.c_str());
             //cout<<parseToken<<endl;
             visualWord.push_back(parseToken);
              }
              visualVocabulary.push_back(visualWord);
         } 
         //cout<< "first "<< visualVocabulary[0][5];
    // Create training data for svm
    ofstream output("train_bow.dat");     
    int count = visualVocabulary.size();
    for(Dataset::const_iterator c_iter=filenames.begin(); c_iter != filenames.end(); ++c_iter)
      { 
        for(int i=0; i<c_iter->second.size(); i++){
          output << classLabels[c_iter->first] << " ";
          int histogram;
          // Compute SIFT for each image and create histogram over 25 visual words.
        CImg<double> input_image(c_iter->second[i].c_str());
        CImg<double> gray = (input_image).get_RGBtoHSI().get_channel(2);
        vector<SiftDescriptor> descriptors = Sift::compute_sift(gray);
        for(int j = 0; j < count-1; j++)
            { histogram = 0;
              for (int i = 0; i < descriptors.size(); i ++)
              {            
              histogram += euclideanDistance(descriptors[i].descriptor, visualVocabulary[j]);             
            } 
              if (histogram != 0)
              output << (j+1) << ":" << histogram << " ";
          }
          // Last visual word.

              histogram = 0;
              for (int i = 0; i < descriptors.size(); i ++)
              { 
              histogram += euclideanDistance(descriptors[i].descriptor, visualVocabulary[count-1]);
            } 
              if (histogram != 0)
              output << count << ":" << histogram;
              output <<endl;
          }
        }
      
      flush(output);
      string str = "./svm_multiclass_learn -c 0.3 -# 300 -t 2 -g 0.00001 train_bow.dat model_bow";
      const char *command = str.c_str();
      system(command);
   }
   // Calculate euclidean distance between two descriptors.
   // Taken from our assignment 2 implementation.
  int euclideanDistance(vector<float> image1_singleDesc, vector<float> image2_singleDesc)  
  { 
  double finalMagnitude = 0.0;
  for(int i=0; i<128; i++)
  {
    finalMagnitude += ((image1_singleDesc[i]) - (image2_singleDesc[i])) * ((image1_singleDesc[i]) - (image2_singleDesc[i]));
  }
  //cout<< "final mag"<< sqrt(finalMagnitude)<<endl;
  if (sqrt(finalMagnitude) <= 275)
    return 1;
  else
    return 0;
}
 
};
