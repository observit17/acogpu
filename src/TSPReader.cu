/****************************************
 * TSPReader.cu                         *
 * Reads .tsp files                     *
 ****************************************/
#include <limits>
#include "TSPReader.h"

TSPReader::TSPReader() {
	cityNames = 0;	
	xcoords = 0;
	ycoords = 0;	
}

//destructor
TSPReader::~TSPReader() {
	if(cityNames != 0)
		delete[] cityNames;
	if(xcoords != 0)
		delete[] xcoords;
	if(ycoords != 0)
		delete[] ycoords;	
}

//read: Reads a given tsp file and extracts data.
bool TSPReader::read(char* filen) {	
	ifstream infile(filen, ios_base::in);
	if(!infile){
		cout << "\n" << "Unable to open file: " << filen << "\n";
		return false;
	}
	string line;
	string tag;
	string value;
	while(infile.good()) {
		getline(infile, line);
		if(line.length() > 1) {
			if(!isprint(line[line.length()-1]))
				line.erase(line.length()-1,1);
			if(line.find(":") != string::npos) {
				tag = line.substr(0,line.find(":"));
				value = line.substr(line.find(":") + 1, line.length() - line.find(":") - 1);
			}
			else {
				tag = line;
				value = "";
			}
			while(tag.find(" ") != string::npos) {
				tag.replace(tag.find(" "),1,"");
			}
			while(value.find(" ") != string::npos) {
				value.replace(value.find(" "),1,"");
			}
			if(tag == "NAME") {
				name = value;
			}
			else if(tag == "TYPE") {
				if(value != "TSP" && value != "STSP"){
					cout << "\n" << "Invalid problem type: " << value << "\n";
					return false;
				}
			}
			else if(tag == "DIMENSION") {
				nodes = atoi(value.c_str());
			}
			else if(tag == "EDGE_WEIGHT_TYPE") {
				if(value != "EUC_2D"){
					cout << "\n" << "Invalid edge weight type: " << value << "\n";
					return false;
				}
			}
			else if(tag == "NODE_COORD_SECTION") {
				//Set coord arrays to appropriate lengths
				cityNames = new string [nodes];
				xcoords = new float [nodes];
				ycoords = new float [nodes];
					
				long int j;
				for(int i = 0; infile.good() && i < nodes; i++) {
					getline(infile,line);
					if(!isprint(line[line.length()-1]))
						line.erase(line.length()-1,1);
					if(line == "EOF") {
						return false;
					}
					cityNames[i] = line.substr(0, line.find(" "));
					//xcoords[i] = atof(line.substr(line.find(" ") + 1, line.find_last_of(" ") - line.find(" ") - 1).c_str());
					//ycoords[i] = atof(line.substr(line.find_last_of(" ") + 1, line.length() - line.find_last_of(" ") - 1).c_str());

					const char* str = line.c_str();	
					sscanf(str,"%ld %f %f", &j, &xcoords[i], &ycoords[i] );
				}
			}
			if(line == "EOF"){
				break;
			}
		}
	}
	
	return true;
}



float* TSPReader::getXcoords() { 
  return xcoords; 
}
	
float* TSPReader::getYcoords() { 
  return ycoords; 
}



string TSPReader::getName() { 
  return name; 
}
int TSPReader::getNumNodes() { 
  return nodes;
}
