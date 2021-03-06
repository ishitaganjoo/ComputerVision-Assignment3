#include <string>

class PCA : public Classifier
{
public:
  PCA(const vector<string> &_class_list) : Classifier(_class_list) {}

  // Principal Component Analysis Training
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

	  //read each vector in a class and convert it into a matrix
	  cout<<"before opening train"<<endl;
	  ofstream myFile;
	  myFile.open("train.dat");
	  int classNum = 0;
	  int c = 0;
	  for(map<int, vector<CImg<double> > >::iterator it = outputMap.begin();it!=outputMap.end();++it)
		{
			classNum++;
			for(int i=0; i<it->second.size(); i++)
			{
				 myFile << classNum <<" ";
				 
				 vector<double> trainVector = eigenDecomposition(it->second[i], c);
				 c++;
				 for(int j = 0; j < trainVector.size(); j++)
				 {
					 myFile << j+1 <<":" << trainVector[j]<<" ";
				 }
				 myFile << endl;
			}
		}

		myFile.flush();
		myFile.close();

		system("./svm_multiclass_learn -c 1.0 train.dat training_model.dat");
  }

  CImg<double> computeCovarianceMatrix(CImg<double> originalMatrix){
    CImg<double> deviationMatrix(size, size);
    CImg<double> covarianceMatrix(size, size);
    CImg<double> covarianceMatrix_Final(size, size);
    CImg<int> oneMatrix(size, size);
    for(int i=0 ;i<size; i++){
      for(int j=0 ; j<size; j++){
        oneMatrix(i, j) = 1;
	       deviationMatrix(i, j) = 0;
         covarianceMatrix(i, j) = 0;
         covarianceMatrix_Final(i, j) = 0;
      }
    }

    for(int i=0; i<size; i++){
      for(int j=0; j<size; j++){
        for(int k=0; k<size; k++){
          deviationMatrix(i, j) += oneMatrix(i, k) * originalMatrix(k, j);
        }
      }
    }
    for(int i=0; i<size; i++){
      for(int j=0; j<size; j++){
        deviationMatrix(i, j) = deviationMatrix(i, j)/size;
      }
    }
    for(int i=0; i<size; i++){
      for(int j=0; j<size; j++){
        covarianceMatrix(i, j) = originalMatrix(i, j) - deviationMatrix(i, j);
      }
    }

    CImg<double> covarianceTranspose = covarianceMatrix.get_transpose();
    for(int i=0; i<size; i++){
      for(int j=0; j<size; j++){
        for(int k=0; k<size; k++){
          covarianceMatrix_Final(i, j) += covarianceTranspose(i, k) * covarianceMatrix(k, j);
        }
      }
    }
    //return covarianceMatrix_Final;
    for(int i=0; i<size; i++){
      for(int j=0; j<size; j++){
        covarianceMatrix_Final(i, j) = covarianceMatrix_Final(i, j)/size;
      }
    }

    return covarianceMatrix_Final;
  }

  //create a new method which calculates the matrix and eigen values and vectors
  vector<double> eigenDecomposition(CImg<double> imageVector, int count)
  {
  	  CImg<double> squareMatrix = imageVector.get_matrix();
    	  CImg<double> covarianceMatrix = computeCovarianceMatrix(squareMatrix);
	  //contains eigen values and eigen vectors
	  CImgList<double> listImages = covarianceMatrix.get_symmetric_eigen();

	  vector<double> trainVector;
	  const char axis= 'x';
	  CImg<double> plotImages(size, 20);
	  for(int i = 0; i<5; i++)
	  {
		  vector<double> sortedMatrix;
		  for(int j = 0; j<size; j++)
		  {
		  	//listImages[0] gives the eigen values and listImages[1] gives the eigen vectors.
		  	//The below code can be uncommented to see the eigen values.
			 //cout<<"eigen values in columns are"<<listImages[0](j,i)<<endl;
			 plotImages(j,i) = listImages[1](j,i);
			 trainVector.push_back(listImages[1](j,i));
		  }
	  }
	  //count++;
	  std::string countFile = std::to_string(count);
	  
// This is the code where the images of the eigen vectors are created. You can uncomment this code to see the images created.
	string fileName = "vectorImage"+countFile+".png";
	  //plotImages.get_normalize(0,255).save(fileName.c_str());

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
			 vector<double> testVector = eigenDecomposition(it->second[i], 0);
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

  static const int size=200;  // subsampled image resolution
  map<string, CImg<double> > models; // trained models
};
