class Haar : public Classifier
{
public:
  Haar(const vector<string> &_class_list) : Classifier(_class_list) {}

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

		vector<CImg<double> > integralImages;
		//vector<CImg<double> > inputImages;

		// convert each image to be a row of this "model" image
		//cout<<"size of image is : "<<c_iter->second.size()<<endl;
		for(int i=0; i<c_iter->second.size(); i++)
		{
		   CImg<double> inputImage = extract_features(c_iter->second[i].c_str());

		   //cout<<"size of image is : "<<(CImg<double>(c_iter->second[i].c_str())).width()<<endl;
		   //cout<<"height is: "<<inputImage.height()<<"width is: "<<inputImage.width()<<endl;
		   //CImgList<double> listImages(2);
		   //listImages[0] = calculateIntegralImage(inputImage);
		   //listImages[1] = inputImage;
		   //inputImages.push_back(inputImage);
		   integralImages.push_back(calculateIntegralImage(inputImage));
		}


		outputMap[++count] = integralImages;

      }

	  vector<CImg<double> > filters;
	  createFilters(&filters);
	  cout<<"size of output map"<<outputMap.size()<<endl;
	  //apply filters on all the integralImages
	  calculateHaarFeatures(filters,outputMap);
	  cout<<"before executing"<<endl;
	  system("./svm_multiclass_learn -c 0.5 train.dat training_model.dat");
      cout<<"after executing"<<endl;

  }

  //create a new method which calculates the integral image
  CImg<double> calculateIntegralImage(CImg<double> inputImage)
  {
	  CImg<double> integralImage(inputImage.width(), inputImage.height());
	  for(int row=0; row<inputImage.height(); row++)
	  {
		  for(int col=0; col<inputImage.width(); col++ )
		  {
			  //check if it is the first row
			  if(row==0)
			  {
				  if(col==0)
				  {
					  integralImage(row, col) = inputImage(row, col);
				  }
				  else
				  {
					  integralImage(row, col) = inputImage(row, col) + integralImage(row, col-1);
				  }
			  }
			  //check if it is the first column
			  else if(col==0)
			  {
				  integralImage(row, col) = inputImage(row, col) + integralImage(row-1, col);

			  }
			  //for all other positions
			  else
			  {
				  integralImage(row, col) = inputImage(row, col) + integralImage(row, col-1) + integralImage(row-1, col) - integralImage(row-1, col-1);
			  }
		  }

	  }
	  //cout<<"IntegralImage size is"<< integralImage.height() << integralImage.width()<< endl;
	  return integralImage;

  }

  void createFilters(vector<CImg<double> > *filters)
  {
	  //create a vector of filters

	  //2*2 filters
	  CImg<double> filter1(2,2);
	  CImg<double> filter2(2,2);
	  CImg<double> filter3(2,2);
	  CImg<double> filter4(2,2);

	  filter1(0,0) = 1;
	  filter1(0,1) = 1;
	  filter1(1,0) = -1;
	  filter1(1,1) = -1;

	  filter2 = filter1.get_rotate(180,1,0);
	  filter3 = filter2.get_rotate(90,1,0);
	  filter4 = filter3.get_rotate(90,1,0);
	  //cout<<"rotated image is :"<<filter2(0,0) <<filter2(0,1) <<filter2(1,0) <<filter2(1,1) <<endl;
	  //cout<<"rotated image is :"<<filter3(0,0) <<filter3(0,1) <<filter3(1,0) <<filter3(1,1) <<endl;
	  //cout<<"rotated image is :"<<filter4(0,0) <<filter4(0,1) <<filter4(1,0) <<filter4(1,1) <<endl;


	  //3*3 filters
	  CImg<double> filter5(3,3);
	  CImg<double> filter6(3,3);
	  CImg<double> filter7(3,3);
	  CImg<double> filter8(3,3);

	  filter5(0,0) = 1;
	  filter5(0,1) = -1;
	  filter5(0,2) = 1;
	  filter5(1,0) = 1;
	  filter5(1,1) = -1;
	  filter5(1,2) = 1;
	  filter5(2,0) = 1;
	  filter5(2,1) = -1;
	  filter5(2,2) = 1;

	  filter6 = filter5.get_rotate(90,1,0);
	  filter7(0,0) = -1;
	  filter7(0,1) = 1;
	  filter7(0,2) = -1;
	  filter7(1,0) = -1;
	  filter7(1,1) = 1;
	  filter7(1,2) = -1;
	  filter7(2,0) = -1;
	  filter7(2,1) = 1;
	  filter7(2,2) = -1;

	  filter8 = filter7.get_rotate(90,1,0);
	  //cout<<"rotated image is :"<<filter5(0,0) <<filter5(0,1) <<filter5(0,2) <<filter5(1,0) <<filter5(1,1)
	  //<<filter5(1,2) <<filter5(2,0) <<filter5(2,1) <<filter5(2,2)<<endl;
	  //cout<<"rotated image is :"<<filter6(0,0) <<filter6(0,1) <<filter6(0,2) <<filter6(1,0) <<filter6(1,1)
	  //<<filter6(1,2) <<filter6(2,0) <<filter6(2,1) <<filter6(2,2)<<endl;
	  //cout<<"rotated image is :"<<filter7(0,0) <<filter7(0,1) <<filter7(0,2) <<filter7(1,0) <<filter7(1,1)
	  //<<filter7(1,2) <<filter7(2,0) <<filter7(2,1) <<filter7(2,2)<<endl;
	  //cout<<"rotated image is :"<<filter8(0,0) <<filter8(0,1) <<filter8(0,2) <<filter8(1,0) <<filter8(1,1)
	  //<<filter8(1,2) <<filter8(2,0) <<filter8(2,1) <<filter8(2,2)<<endl;
	  //cout<<"rotated image is :"<<filter3(0,0) <<filter3(0,1) <<filter3(1,0) <<filter3(1,1) <<endl;
	  //cout<<"rotated image is :"<<filter4(0,0) <<filter4(0,1) <<filter4(1,0) <<filter4(1,1) <<endl;
	  filters->push_back(filter1);
	  filters->push_back(filter2);
      filters->push_back(filter3);
	  filters->push_back(filter4);
	  filters->push_back(filter5);
	  filters->push_back(filter6);
	  filters->push_back(filter7);
	  filters->push_back(filter8);
  }

  void calculateHaarFeatures(vector<CImg<double> > filters, map<int, vector<CImg<double> > > outputMap)
  {
	  ofstream myFile;
	  myFile.open("train.dat");
	  int classNum = 0;
	  for(map<int, vector<CImg<double> > >::iterator it = outputMap.begin();it!=outputMap.end();++it)
		{
			classNum++;
			for(int i=0; i<it->second.size(); i++)
			{
				//cout<<"ishita "<<it->second.size()<<endl;
				myFile << classNum <<" ";
				vector<double> imageDescriptor = applyFilter4(it->second[i], filters);
        vector<double> imageDescriptor_2 = applyFilter(it->second[i], filters);
        //for(int j = 0; j<imageDescriptor_2.size(); j++){
        //  imageDescriptor.push_back(imageDescriptor_2[j]);
       // }
				for(int j = 0; j<imageDescriptor.size(); j++)
				{
					myFile << j+1 <<":" << imageDescriptor[j]<<" ";
				}
				myFile << endl;
			}

		}
		myFile.flush();
		myFile.close();


  }

    void calculateHaarFeatures_2(vector<CImg<double> > filters, map<int, vector<CImg<double> > > outputMap, int actualClass)
  {
	  ofstream myFile;
	  myFile.open("test.dat");
	  int classNum = 0;
	  for(map<int, vector<CImg<double> > >::iterator it = outputMap.begin();it!=outputMap.end();++it)
		{
			classNum++;
			for(int i=0; i<it->second.size(); i++)
			{
				//cout<<"ishita "<<it->second.size()<<endl;
				myFile << actualClass <<" ";
				vector<double> imageDescriptor = applyFilter4(it->second[i], filters);
        vector<double> imageDescriptor_2 = applyFilter(it->second[i], filters);
        //for(int j = 0; j<imageDescriptor_2.size(); j++){
         // imageDescriptor.push_back(imageDescriptor_2[j]);
       // }
				for(int j = 0; j<imageDescriptor.size(); j++)
				{
					myFile << j+1 <<":" << imageDescriptor[j]<<" ";
				}
				myFile << endl;
			}

		}
		myFile.flush();
		myFile.close();


  }

  vector<double> applyFilter(CImg<double> integralImage, vector<CImg<double> > filters)
  {
	  vector<double> imageDescriptor;
	  for(int row=0; row<integralImage.height(); row+=2)
	  {
		  for(int col=0; col<integralImage.width(); col+=2)
		  {
			  //apply all filters
			  for(int i=0; i<filters.size(); i++)
			  {
				  double whiteSum = 0;
				  double blackSum = 0;
				  if(i==0)
				  {
					  int x1 = row+filters[0].height()-1;
					  int y1 = col;
					  double d = (x1<0 || y1<0) ? 0 : integralImage(x1,y1);


					  int x2 = x1 - filters[0].height();
					  int y2 = y1 - 1;
					  double a = (x2<0 || y2<0) ? 0 : integralImage(x2,y2);

					  int x3 = x2;
					  int y3 = y1;
					  double b = (x3<0 || y3<0) ? 0 : integralImage(x3,y3);

					  int x4 = x1;
					  int y4 = y2;
					  double c = (x4<0 || y4<0) ? 0 : integralImage(x4,y4);

					  whiteSum = d + a - b - c;
					  //cout<<"white sum is "<<whiteSum<<endl;

					  int x5 = x1;
					  int y5 = y1+1;
					  d = (x5<0 || y5<0) ? 0 : integralImage(x5,y5);


					  int x6 = x5 - filters[0].height();
					  int y6 = y5 - 1;
					  a = (x6<0 || y6<0) ? 0 : integralImage(x6,y6);

					  int x7 = x6;
					  int y7 = y5;
					  b = (x7<0 || y7<0) ? 0 : integralImage(x7,y7);

					  int x8 = x5;
					  int y8 = y6;
					  c = (x8<0 || y8<0) ? 0 : integralImage(x8,y8);

					  blackSum = d + a - b - c;
					  //cout<<"blackSum is "<<blackSum<<endl;
					  imageDescriptor.push_back(blackSum-whiteSum);
					  imageDescriptor.push_back(whiteSum-blackSum);
				  }

				  else if(i==1)
				  {
					  int x1 = row+filters[0].height()-1;
					  int y1 = col + 1;
					  double d = (x1<0 || y1<0) ? 0 : integralImage(x1,y1);


					  int x2 = row;
					  int y2 = col - 1;
					  double a = (x2<0 || y2<0) ? 0 : integralImage(x2,y2);

					  int x3 = x1-1;
					  int y3 = y1;
					  double b = (x3<0 || y3<0) ? 0 : integralImage(x3,y3);

					  int x4 = x1;
					  int y4 = y2;
					  double c = (x4<0 || y4<0) ? 0 : integralImage(x4,y4);

					  blackSum = d + a - b - c;
					  //cout<<"blackSum is "<<blackSum<<endl;

					  int x5 = row;
					  int y5 = y1;
					  d = (x5<0 || y5<0) ? 0 : integralImage(x5,y5);


					  int x6 = x5 - 1;
					  int y6 = y2;
					  a = (x6<0 || y6<0) ? 0 : integralImage(x6,y6);

					  int x7 = x6;
					  int y7 = y5;
					  b = (x7<0 || y7<0) ? 0 : integralImage(x7,y7);

					  int x8 = x5;
					  int y8 = y6;
					  c = (x8<0 || y8<0) ? 0 : integralImage(x8,y8);

					  whiteSum = d + a - b - c;
					  //cout<<"whiteSum is "<<whiteSum<<endl;
					  imageDescriptor.push_back(blackSum-whiteSum);
					  imageDescriptor.push_back(whiteSum-blackSum);

				  }
				  // else if(i==2)
				  // {
					//   blackSum = originalImage(row, col) + originalImage(row+1, col+1);
					//   whiteSum = originalImage(row, col+1) + originalImage(row+1, col);
					//   imageDescriptor.push_back(blackSum - whiteSum);
					//   imageDescriptor.push_back(whiteSum - blackSum);
          //
				  // }

			  }

		  }

	  }
   // for(int i=0; i<imageDescriptor.size(); i++){
   //   cout<<"imageDescriptor "<<imageDescriptor[i];
    //}
    return imageDescriptor;
  }

  vector<double> applyFilter4(CImg<double> integralImage, vector<CImg<double> > filters)
  {
	  vector<double> imageDescriptor;
	  for(int row=0; row<integralImage.height(); row+=4)
	  {
		  for(int col=0; col<integralImage.width(); col+=4)
		  {
			  //apply all filters
			  for(int i=0; i<filters.size(); i++)
			  {
				  double whiteSum = 0;
				  double blackSum = 0;
				  if(i==0)
				  {
					  int x1 = row+filters[0].height()-1;
					  int y1 = col+1;
					  double d = (x1<0 || y1<0) ? 0 : integralImage(x1,y1);


					  int x2 = x1 - filters[0].height();
					  int y2 = y1 - 2;
					  double a = (x2<0 || y2<0) ? 0 : integralImage(x2,y2);

					  int x3 = x2;
					  int y3 = y1;
					  double b = (x3<0 || y3<0) ? 0 : integralImage(x3,y3);

					  int x4 = x1;
					  int y4 = y2;
					  double c = (x4<0 || y4<0) ? 0 : integralImage(x4,y4);

					  whiteSum = d + a - b - c;
					  //cout<<"white sum is "<<whiteSum<<endl;

					  int x5 = x1;
					  int y5 = y1+2;
					  d = (x5<0 || y5<0) ? 0 : integralImage(x5,y5);


					  int x6 = x5 - filters[0].height();
					  int y6 = y5 - 2;
					  a = (x6<0 || y6<0) ? 0 : integralImage(x6,y6);

					  int x7 = x6;
					  int y7 = y5;
					  b = (x7<0 || y7<0) ? 0 : integralImage(x7,y7);

					  int x8 = x5;
					  int y8 = y6;
					  c = (x8<0 || y8<0) ? 0 : integralImage(x8,y8);

					  blackSum = d + a - b - c;
					  //cout<<"blackSum is "<<blackSum<<endl;
					  imageDescriptor.push_back(blackSum-whiteSum);
					  imageDescriptor.push_back(whiteSum-blackSum);
				  }

				  else if(i==1)
				  {
					  int x1 = row+filters[0].height()-1;
					  int y1 = col + 3;
					  double d = (x1<0 || y1<0) ? 0 : integralImage(x1,y1);


					  int x2 = row+1;
					  int y2 = col - 1;
					  double a = (x2<0 || y2<0) ? 0 : integralImage(x2,y2);

					  int x3 = x2;
					  int y3 = y1;
					  double b = (x3<0 || y3<0) ? 0 : integralImage(x3,y3);

					  int x4 = x1;
					  int y4 = y2;
					  double c = (x4<0 || y4<0) ? 0 : integralImage(x4,y4);

					  blackSum = d + a - b - c;
					  //cout<<"blackSum is "<<blackSum<<endl;

					  int x5 = x3;
					  int y5 = y3;
					  d = (x5<0 || y5<0) ? 0 : integralImage(x5,y5);


					  int x6 = row - 1;
					  int y6 = y2;
					  a = (x6<0 || y6<0) ? 0 : integralImage(x6,y6);

					  int x7 = x6;
					  int y7 = y5;
					  b = (x7<0 || y7<0) ? 0 : integralImage(x7,y7);

					  int x8 = x5;
					  int y8 = y6;
					  c = (x8<0 || y8<0) ? 0 : integralImage(x8,y8);

					  whiteSum = d + a - b - c;
					  //cout<<"whiteSum is "<<whiteSum<<endl;
					  imageDescriptor.push_back(blackSum-whiteSum);
					  imageDescriptor.push_back(whiteSum-blackSum);

				  }
				  else if(i==2)
				  {
					  int x1 = row+filters[0].height()-1;
					  int y1 = col + 3;
					  double d = (x1<0 || y1<0) ? 0 : integralImage(x1,y1);


					  int x2 = row+1;
					  int y2 = col + 1;
					  double a = (x2<0 || y2<0) ? 0 : integralImage(x2,y2);

					  int x3 = x2;
					  int y3 = y1;
					  double b = (x3<0 || y3<0) ? 0 : integralImage(x3,y3);

					  int x4 = x1;
					  int y4 = y2;
					  double c = (x4<0 || y4<0) ? 0 : integralImage(x4,y4);

					  double blackSum1 = d + a - b - c;
					  //cout<<"blackSum is "<<blackSum1<<endl;

					  x1 = row + 1;
					  y1 = col + 1;
					  d = (x1<0 || y1<0) ? 0 : integralImage(x1,y1);


					  x2 = row - 1;
					  y2 = col - 1;
					  a = (x2<0 || y2<0) ? 0 : integralImage(x2,y2);

					  x3 = x2;
					  y3 = y1;
					  b = (x3<0 || y3<0) ? 0 : integralImage(x3,y3);

					  x4 = x1;
					  y4 = y2;
					  c = (x4<0 || y4<0) ? 0 : integralImage(x4,y4);

					  double blackSum2 = d + a - b - c;
					  //cout<<"blackSum is "<<blackSum2<<endl;

					  blackSum = blackSum1 + blackSum2;

					  int x5 = row + 1;
					  int y5 = y3;
					  d = (x5<0 || y5<0) ? 0 : integralImage(x5,y5);


					  int x6 = x3;
					  int y6 = y3;
					  a = (x6<0 || y6<0) ? 0 : integralImage(x6,y6);

					  int x7 = x6;
					  int y7 = col + 3;
					  b = (x7<0 || y7<0) ? 0 : integralImage(x7,y7);

					  int x8 = x1;
					  int y8 = y1;
					  c = (x8<0 || y8<0) ? 0 : integralImage(x8,y8);

					  double whiteSum1 = d + a - b - c;
					  //cout<<"whiteSum is "<<whiteSum1<<endl;

					  x5 = row + 3;
					  y5 = col + 1;
					  d = (x5<0 || y5<0) ? 0 : integralImage(x5,y5);


					  x6 = row + 1;
					  y6 = col - 1;
					  a = (x6<0 || y6<0) ? 0 : integralImage(x6,y6);

					  x7 = row + 1;
					  y7 = col + 1;
					  b = (x7<0 || y7<0) ? 0 : integralImage(x7,y7);

					  x8 = row + 3;
					  y8 = col - 1;
					  c = (x8<0 || y8<0) ? 0 : integralImage(x8,y8);

					  double whiteSum2 = d + a - b - c;
					  //cout<<"whiteSum is "<<whiteSum2<<endl;

					  whiteSum = whiteSum1 + whiteSum2;

					  imageDescriptor.push_back(blackSum-whiteSum);
					  imageDescriptor.push_back(whiteSum-blackSum);

				  }

			  }

		  }

	  }
	  return imageDescriptor;
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
    CImg<double> test_image = extract_features(filename);
	cout<<"name "<<filename<<endl;

	vector<CImg<double> > integralImage(1);

	integralImage[0] = calculateIntegralImage(test_image);
	map<int, vector<CImg<double> > > outputMap;
	outputMap[actualClass] = integralImage;
	vector<CImg<double> > filters;
	  createFilters(&filters);
	  //cout<<"size of output map"<<outputMap.size()<<endl;
	  //apply filters on all the integralImages
	  calculateHaarFeatures_2(filters,outputMap, actualClass);
	  //cout<<"before executing"<<endl;
	  system("./svm_multiclass_classify test.dat training_model.dat predictions.dat");
    string line;
    ifstream file;
    file.open("predictions.dat");
    //cout <<"ORIGINAL IS "<<actualClass<<endl;
    while(getline(file, line)) {
      cout<< "PREDICTION IS "<<line<<endl;
      int p = atoi(line.c_str());
    return classLabels[p];
  //     //cout<<"after executing"<<endl;
  //     	//exit(0);
	//
	// //calculateHaarFeatures(filters,outputMap);
  //   // figure nearest neighbor
  //   pair<string, double> best("", 10e100);
  //   double this_cost;
  //   for(int c=0; c<class_list.size(); c++)
  //     for(int row=0; row<models[ class_list[c] ].height(); row++)
	// if((this_cost = (test_image - models[ class_list[c] ].get_row(row)).magnitude()) < best.second)
	//   best = make_pair(class_list[c], this_cost);
  //
  //   return best.first;
  }
}

  virtual void load_model()
  {
    for(int c=0; c < class_list.size(); c++){}
     // models[class_list[c] ] = (CImg<double>(("nn_model." + class_list[c] + ".png").c_str()));
  }
protected:
  // extract features from an image, which in this case just involves resampling and
  // rearranging into a vector of pixel data.
  CImg<double> extract_features(const string &filename)
    {
      return (CImg<double>(filename.c_str())).get_RGBtoHSI().get_channel(2).resize(size,size,1,1,3);
    }

  static const int size=256;  // subsampled image resolution
  map<string, CImg<double> > models; // trained models
};
