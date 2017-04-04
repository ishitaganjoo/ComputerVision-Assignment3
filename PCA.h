class PCA : public Classifier
{
public:
  PCA(const vector<string> &_class_list) : Classifier(_class_list) {}
  
  // Nearest neighbor training. All this does is read in all the images, resize
  // them to a common size, convert to greyscale, and dump them as vectors to a file
  virtual void train(const Dataset &filenames) 
  {
	map<int, vector<CImg<double> > > outputMap; 
	int count = 0;
    for(Dataset::const_iterator c_iter=filenames.begin(); c_iter != filenames.end(); ++c_iter)
      {
		cout << "Processing " << c_iter->first << endl;
		CImg<double> class_vectors(size*size, filenames.size(), 1);
		
		vector<CImg<double> > imageVectors;
		
		// convert each image to be a row of this "model" image
		for(int i=0; i<c_iter->second.size(); i++) 
		{
		   CImg<double> features = extract_features(c_iter->second[i].c_str());
		   imageVectors.push_back(features);
		}
	  
		
		outputMap[++count] = imageVectors;
	
      }
	  
	  //cout<<"map size is"<<outputMap.size()<<endl;
	  //read each vector in a class and convert it into a matrix
	  cout<<"before opening train"<<endl;
	  ofstream myFile;
	  myFile.open("train.dat");
	  int classNum = 0;
	  for(map<int, vector<CImg<double> > >::iterator it = outputMap.begin();it!=outputMap.end();++it)
		{
			classNum++;
			for(int i=0; i<it->second.size(); i++)
			{
				 myFile << classNum <<" ";
				 vector<double> trainVector = eigenDecomposition(it->second[i]);
				 //cout<<"test vector size is "<<trainVector.size()<<endl;
				 for(int j = 0; j < trainVector.size(); j++)
				 {
					 myFile << j+1 <<":" << trainVector[j]<<" ";
				 }
				 myFile << endl;
			}
		}
		
		//cout<<"outside for "<<endl;
		myFile.flush();
		myFile.close();
		
		system("./svm_multiclass_learn -c 1.0 train.dat training_model.dat");
  }
  
  //create a new method which calculates the matrix and eigen values and vectors
  vector<double> eigenDecomposition(CImg<double> imageVector)
  {
	  
	  CImg<double> squareMatrix = imageVector.get_matrix();
	  //contains eigen values and eigen vectors
	  CImgList<double> listImages = squareMatrix.get_symmetric_eigen();
	  
	  //CImg<double> class_vectors(size*size, 1, 1);
      //cout<<"eigen vector width"<<listImages[1].height() << listImages[1].width()<<endl;
	  vector<double> trainVector;
	  const char axis= 'x';
	  listImages[1].sort(false, axis);
	  for(int i = 0; i<size; i++)
	  {
		  vector<double> sortedMatrix;
		  for(int j = 0; j<size; j++)
		  {
			 //cout<<"vectors in columns are"<<listImages[1](j,i)<<endl;
			 sortedMatrix.push_back(listImages[1](j,i));
			 //trainVector.push_back(listImages[1](j,i));
		  }
		  //cout<<"next column"<<endl;
		  sort(sortedMatrix.begin(), sortedMatrix.end());
		  for(int k=0; k<200; k++)
		  {
			  //cout<<"values are"<<sortedMatrix[k]<<endl;
			  trainVector.push_back(sortedMatrix[k]);
		  }
		  //break;
	  }
	  //choose top k eigen vectors
	  //class_vectors = class_vectors.draw_image(0, 0, 0, 0, testVector);
	
	  //class_vectors.save_png(("nn_model. eigenVectorImage .png"));
	  //CImg<double> eigenVectors = listImages[1].get_vector();
	  
	  return trainVector;
  }
  
  virtual string classify(const string &filename, int actualClass)
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
	map<int, vector<CImg<double> > > outputMap;
    CImg<double> test_image = extract_features(filename);
	vector<CImg<double> > imageVectors;
	imageVectors.push_back(test_image);
	outputMap[actualClass] = imageVectors;    

	
	ofstream myFile;
	myFile.open("test_pca.dat");
	
	for(map<int, vector<CImg<double> > >::iterator it = outputMap.begin();it!=outputMap.end();++it)
	{
		
		for(int i=0; i<it->second.size(); i++)
		{
			 myFile << actualClass <<" ";
			 vector<double> testVector = eigenDecomposition(it->second[i]);
			 cout<<"test vector size is "<<testVector.size()<<endl;
			 for(int j = 0; j < testVector.size(); j++)
			 {
				 myFile << j+1 <<":" << testVector[j]<<" ";
			 }
			 myFile << endl;
		}
	}
	myFile.flush();
	myFile.close();
	system("./svm_multiclass_classify test_pca.dat training_model.dat predictions.dat");
    string line;
    ifstream file;
    file.open("predictions.dat");
    //cout <<"ORIGINAL IS "<<actualClass<<endl;
    while(getline(file, line)) {
      cout<< "PREDICTION IS "<<line<<endl;
      int p = atoi(line.c_str());
	  cout<<"p is "<< p<< endl;
    return classLabels[p];
	}
  }
  

  virtual void load_model()
  {
    for(int c=0; c < class_list.size(); c++){}
      //models[class_list[c] ] = (CImg<double>(("nn_model." + class_list[c] + ".png").c_str()));
  }
protected:
  // extract features from an image, which in this case just involves resampling and 
  // rearranging into a vector of pixel data.
  CImg<double> extract_features(const string &filename)
    {
      return (CImg<double>(filename.c_str())).get_RGBtoHSI().get_channel(2).resize(size,size,1,1,3).unroll('x');
    }

  static const int size=500;  // subsampled image resolution
  map<string, CImg<double> > models; // trained models
};
