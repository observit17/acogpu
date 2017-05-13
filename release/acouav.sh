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
	tsp=$2
	optimum=$3

	#for ants in $cities, 1024, 2048, 4096
        for ants in 1024, 2048, 4096
	do
		iter=10
		while [ $iter -le 100 ]
		do
			if [ -f "$EXEACO_GPU" ]; then
				$EXEACO_GPU -m $ants -s $iter -o $optimum -t $MAXTIME -i $TSPDIR/$tsp.tsp -p $OUTDIR/$tsp.txt -a 1.0 -b 2.0 -r 0.5		
			fi
			#if [ -f "$EXEACO_CPU" ]; then
			#	$EXEACO_CPU -m $ants -s $iter -o $optimum -t $MAXTIME -i $TSPDIR/$tsp.tsp -p $OUTDIR/$tsp.txt -a 1.0 -b 2.0 -r 0.5
			#fi
			
			iter=`expr $iter + 10`
		done
	done
}

#
#
# berlin52.tsp
#
#EXEC_ACOUAV 52 berlin52 7542 

#
# eil76.tsp
#
#EXEC_ACOUAV 76 eil76 538 

#
# kroA100.tsp
#
#EXEC_ACOUAV 100 kroA100 21282 

#
# tsp225.tsp
#
#EXEC_ACOUAV 225 tsp225 3916 

#
# pr439.tsp
#
#EXEC_ACOUAV 439 pr439 107217 

#
# pr1002.tsp
#
EXEC_ACOUAV 1002 pr1002 259045 

# end
