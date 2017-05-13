#
# Execute acouav_{cpu/gpu}
#
#

# The executable path of this file
# Use the absolute path if running from some other directory
ACOUAV_HOME=.


# max time to execute a trial (for one iteration) in millis
# this value may needed to bound CPU running passes
# 1 hour = 3600000 milis
MAXTIME=3600000

EXEACO_CPU=$ACOUAV_HOME/acouav_cpu
EXEACO_GPU=$ACOUAV_HOME/acouav_gpu

NOW=$(date +"%m%d%Y%H%M%S")

TSPDIR=$ACOUAV_HOME/input
OUTDIR=$ACOUAV_HOME/output/$NOW

# create output directory if not exists
mkdir -p $OUTDIR

if [ !$MAXTIME ]; then
	# set to 24 hours per iteration
	MAXTIME=43200000
fi

EXEC_ACOUAV() {
	cities=$1
	ants=$2
	tsp=$3
	optimum=$4
	iter=$5
	
	alpha=1
	
	while [ $alpha -le 5 ]
	#for alpha in 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0
	do
		beta=1
		#for beta in 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0
		while [ $beta -le 5 ]
		do
			if [ -f "$EXEACO_GPU" ]; then
				$EXEACO_GPU -m $ants -s $iter -o $optimum -t $MAXTIME -i $TSPDIR/$tsp.tsp -p $OUTDIR/$tsp.txt -a $alpha -b $beta -r 0.5		
			fi
			#if [ -f "$EXEACO_CPU" ]; then
			#	$EXEACO_CPU -m $ants -s $iter -o $optimum -t $MAXTIME -i $TSPDIR/$tsp.tsp -p $OUTDIR/$tsp.txt -a 1.0 -b 2.0 -r 0.5
			#fi
			
			beta=`expr $beta + 1`
		done
		
		alpha=`expr $alpha + 1`
	done
}

#
#
# berlin52.tsp
#
EXEC_ACOUAV 52 4096 berlin52 7542 100
#
# eil76.tsp
#
EXEC_ACOUAV 76 4096 eil76 538 100 
#
# kroA100.tsp
#
EXEC_ACOUAV 100 4096 kroA100 21282 100 
#
# tsp225.tsp
#
EXEC_ACOUAV 225 4096 tsp225 3916 100 
#
# pr439.tsp
#
EXEC_ACOUAV 439 4096 pr439 107217 100 
#
# pr1002.tsp
#
EXEC_ACOUAV 1002 4096 pr1002 259045 10 

# end
