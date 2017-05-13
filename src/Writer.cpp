

#include "Writer.h"
#include <string.h>
#include <iomanip>

const int w = 12;
const char sep = ',';


const char EXT_LOG[5] = ".log";
const char EXT_CSV[5] = ".csv";

bool Writer::VERBOSE = false;

inline bool file_empty (const std::string& name) {
	bool ret = true;
    ifstream f(name.c_str());
    if (f.good()) {
    	ret = f.peek() == std::ifstream::traits_type::eof();
        f.close();
    } else {
        f.close();
        ret = true;
    }
    return ret;
}

//Constructor: Sets defaults.
Writer::Writer() : writing(false) {	
}

//Constructor: Sets defaults and opens the given file.
Writer::Writer(const char* filen) : writing(false) {
	
	setFile(filen);
}

//Constructor: Sets defaults and opens the given file.
Writer::Writer(const char* fname, const char* ext) : writing(false) {
	
	if(fname != NULL && fname[0] != '\0') {
		char filen[MAX_INPUT_LEN];
		strcpy(filen, fname);
		strcat(filen, ext);
		setFile(filen);
	}

}

Writer::~Writer()//Destructor.
{
	close();
}


//setFile: Tries to open given file. If it does, it is changed to writing mode.
bool Writer::setFile(const char* fn) {  

	if( fn != NULL && fn[0] != '\0' ) {
		filename = fn;
	
		writing = true;
		f.open(fn, ios_base::app);
		f <<  setprecision(OUTPUT_PRECISION);
		if (!f){
			writing = false;
		}
	}
	return writing;
}

void Writer::close() {
	if(writing) {
		f.close();
	}
	writing = false;
}

//writeHeader: Writes a header to the file and (if in writing mode) to the file.
void Writer::writeHeader(const char* platform, float alpha, float beta, float rho, int numAnts, string ACO, string TSPName, float calcdisttime)
{
  time_t rawtime;
  time ( &rawtime );
  if(writing){
    f << "\n" << "Date: " << ctime (&rawtime) <<
		"PLATFORM: " << platform << "\n" <<
		"TSP: " << TSPName.c_str() << "\n" <<
		"ACO: " << ACO.c_str() << "\n" <<
		"Ants: " << numAnts << "\n" << 
		"Ants Aging: " << (ANTS_AGING ? "TRUE" : "FALSE") << "\n" <<
		"Distance calculation time: " << calcdisttime << "\n" <<
		"Alpha: " << alpha << " Beta: " << beta << " Rho: " << rho << "\n" <<
		std::left << setw(w) << "Iter#"<< setw(w) << "#Ants"<< setw(w)  << "I-Best" << setw(w) << "G-Best" << setw(w) << "I-Time" << setw(w) << "T-Time" << setw(w) <<
		"Init Tour" << setw(w) << "Tour" << setw(w*2) << "Compute Ant Distances" << setw(w*2) <<
		"Evaporate Pheromones" << setw(w*2) << "Update Pheromones" << setw(w*2) << "Compute Probabilities" <<
		std::endl << std::flush;
  }
  if(Writer::VERBOSE) {
  cout << "\n" << "Date: " << ctime (&rawtime) <<
    "TSP: " << TSPName.c_str() << "\n" <<
		"ACO: " << ACO.c_str() << "\n" <<
		"Ants: " << numAnts << "\n" << 
		"Distance calculation time: " << calcdisttime << "\n" <<
		"Alpha: " << alpha << " Beta: " << beta << " Rho: " << rho << "\n" <<
		std::left << setw(w) << "Iter#"<< setw(w) << "#Ants"<< setw(w) << "I-Best" << setw(w) << "G-Best" << setw(w) << "I-Time" << setw(w) << "T-Time" <<
#if 0
		setw(w*3) <<
		"Init Tour" << setw(w*3) << "Tour" << setw(w*3) << "Compute Ant Distances" << setw(w*3) << 
		"Evaporate Pheromones" << setw(w*3) << "Update Pheromones" << setw(w*3) << "Compute Probabilities" << std::endl << 
		std::left << setw(w*5) << "" << setw(w) <<
		"CPU" << setw(w) << "GPU" << setw(w) << "Total" << setw(w) <<
		"CPU" << setw(w) << "GPU" << setw(w) << "Total" << setw(w) <<
		"CPU" << setw(w) << "GPU" << setw(w) << "Total" << setw(w) <<
		"CPU" << setw(w) << "GPU" << setw(w) << "Total" << setw(w) <<
		"CPU" << setw(w) << "GPU" << setw(w) << "Total" << setw(w) <<
		"CPU" << setw(w) << "GPU" << setw(w) << "Total" <<
#endif
		std::endl;
  }
#if 0
  else {
	  std::cout << std::left << setw(w) << "platform" << setw(w) << "cities" << setw(w) << "ants" << setw(w) << "itererations" << setw(w) <<
#if 0
	  		  "calc-dist-time" << setw(w) << "init-pheromones" << setw(w) << "evap-pheromnes" << setw(w) <<
	    		  "cpu-time" << setw(w) << "gpu-time" << setw(w) <<
#endif
	    		  "total-time" << setw(w) <<
#if 0
	    		  "optimal-distance" << setw(w) << "calc-distance" << setw(w) <<
#endif
	    		  "error %" <<
	    		  "\n";
  }
#endif
}

//write: Writes a standard line of output to stdout and (if in writing mode) to the file.
void Writer::write(int iter, float iterBest, float globBest, float iterTime, float totalTime)
{

  if(writing){
    //f << iter << setw(w) << iterBest << "\t" << globBest << "\t" << iterTime << "\t" << time << "\n" << flush;
	  f << std::left << setw(w) << iter << setw(w) << iterBest << setw(w) << globBest << setw(w) << iterTime << setw(w) << totalTime << "\n" << flush;
  }
  if(Writer::VERBOSE) {
  cout << std::left << setw(w) << iter << setw(w) << iterBest << setw(w) << globBest << setw(w) << iterTime << setw(w) << totalTime << "\n";
  }
}

void Writer::write(const char* dump) {
	if(writing){
		f << dump << endl << flush;
	}
	//cout << dump << endl;
}

void Writer::write(const Statistics& statistics) {
	
	 if(writing){
		 f << std::left << setw(w) << 
			statistics.iter << setw(w) <<
			statistics.ants  << setw(w) << 
			statistics.iterbest << setw(w) << 
			statistics.globalbest << setw(w) <<
			statistics.t_itertime  << setw(w) << 			
			statistics.t_globaltime << setw(w) <<
			statistics.t_inittour << setw(w) <<
			statistics.t_tour << setw(w*2) <<
			statistics.t_antdist << setw(w*2) <<
			statistics.t_evapphoremone << setw(w*2) <<
			statistics.t_updatephoremone << setw(w*2) <<
			statistics.t_computeprob <<
			"\n" << flush;
	 }
	 if(Writer::VERBOSE) {
		std::cout << std::left << setw(w) << statistics.iter << setw(w) << statistics.ants << setw(w) << statistics.iterbest << setw(w) << statistics.globalbest << setw(w) <<
			statistics.t_itertime << setw(w) << statistics.t_globaltime <<
#if 0
			setw(w)<<
			statistics.inittourcpu<<setw(w)<<statistics.inittourgpu<<setw(w)<<statistics.inittourcpu+statistics.inittourgpu<<setw(w)<<
			statistics.tourcpu<<setw(w)<<statistics.tourgpu<<setw(w)<<statistics.tourcpu+statistics.tourgpu<<setw(w)<<
			statistics.antdistcpu<<setw(w)<<statistics.antdistgpu<<setw(w)<<statistics.antdistcpu+statistics.antdistgpu<<setw(w)<<
			statistics.evapphoremonecpu<<setw(w)<<statistics.evapphoremonegpu<<setw(w)<<statistics.evapphoremonecpu+statistics.evapphoremonegpu<<setw(w)<<
			statistics.updatephoremonecpu<<setw(w)<<statistics.updatephoremonegpu<<setw(w)<<statistics.updatephoremonecpu+statistics.updatephoremonegpu<<setw(w)<<
			statistics.computeprobcpu<<setw(w)<<statistics.computeprobgpu<<setw(w)<<statistics.computeprobcpu+statistics.computeprobgpu<<
#endif
			"\n";
	 }
}

void Writer::writelogheader() {
	if(writing && file_empty(filename)) {
		f << std::left << setw(w) << "platform" << setw(w) << "cities" << setw(w) << "ants" << setw(w) << "iters" << setw(w) <<
			//"aging" << setw(w) << "alpha" << setw(w) << "beta" << setw(w) <<
	  		  "init-t" << setw(w) << "caldist-t" << setw(w) << 
			  "init-tour" << setw(w) << "comp-prob" << setw(w) << "tour-time" << setw(w) << "dist-calc" << setw(w) <<
			  "evap-pher" << setw(w) << "update-pher" << setw(w) <<	"total-time" << setw(w) <<
	    		  "opt-dist" << setw(w) << "calc-dist" << setw(w) <<
	    		  "error %" <<
	    		  "\n" << flush;
	}
}
void Writer::writelog(const SummaryStatistics& summary) {

  if(writing) {
	  f << std::left << setw(w) << summary.platform << setw(w) << summary.numcities << setw(w) << summary.numants << setw(w) << summary.numiter << setw(w) <<
		  //(ANTS_AGING ? "TRUE" : "FALSE") << setw(w) << summary.p_alpha << setw(w) << summary.p_beta << setw(w) << 
		  summary.t_inittime << setw(w) << summary.t_distcalc << setw(w) << 
		  summary.t_inittour << setw(w) << summary.t_computeprob << setw(w) << summary.t_tour << setw(w) << summary.t_antdist << setw(w) <<
		  summary.t_evappheromnes << setw(w) << summary.t_updatephoremone << setw(w) << summary.t_totaltime << setw(w) <<
		  summary.d_optimal << setw(w) << summary.d_calulated << setw(w) << summary.d_error <<
		  "\n" << flush;
  }

  if(!Writer::VERBOSE) {
  std::cout << std::left << setw(w) << summary.platform << setw(w) << summary.numcities << setw(w) << summary.numants << setw(w) << summary.numiter << setw(w) <<
#if 0
		  summary.disttime << setw(w) << summary.initpheromones << setw(w) << summary.evappheromnes << setw(w) <<
  		  summary.cputime << setw(w) << summary.gputime << setw(w) <<
#endif
  		  summary.t_totaltime << setw(w) <<
#if 0
  		  summary.optdistance << setw(w) << summary.caldistance << setw(w) <<
#endif
  		  summary.d_error <<
  		  "\n";
  }
 
}
void Writer::writecsvheader() {
	if(writing && file_empty(filename)) {
		f << "platform" << sep << "cities" << sep << "ants" << sep << "itererations" << sep <<
			"aging" << sep << "alpha" << sep << "beta" << sep <<
	  		  "init-time" << sep << "calc-dist-time" << sep << 
			  "init-tour" << sep << "comp-prob" << sep << "tour-time" << sep << "dist-calc" << sep <<
			  "evap-pheromones" << sep << "update-pheromnes" << sep <<
	    		  "total-time" << sep <<
	    		  "optimal-distance" << sep << "calc-distance" << sep <<
	    		  "error %" <<
	    		  "\n" << flush;
	}
}
void Writer::writecsv(const SummaryStatistics& summary) {

  if(writing) {
	  f << summary.platform << sep << summary.numcities << sep << summary.numants << sep << summary.numiter << sep <<
		  (ANTS_AGING ? "TRUE" : "FALSE") << sep << summary.p_alpha << sep << summary.p_beta << sep <<
		  summary.t_inittime << sep << summary.t_distcalc << sep << 
		   summary.t_inittour << sep << summary.t_computeprob << sep << summary.t_tour << sep << summary.t_antdist << sep <<
		  summary.t_evappheromnes << sep << summary.t_updatephoremone << sep << summary.t_totaltime << sep <<
		  summary.d_optimal << sep << summary.d_calulated << sep << summary.d_error <<
		  "\n" << flush;
  }
 
}

