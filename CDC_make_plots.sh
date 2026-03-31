# Define Log Files
expander_contractor='20260329_160146'
compass_gait_walker='20260328_211440'

# Make Hero Figure Plot (Fig 1)
python Plotting/PlotAlgorithmSteps.py 

# Make Algorithm Steps Plots (Fig 2a, 2b)
python Plotting/PlotAlgorithmSets.py Data/CompassGaitWalker_ellipsoid_log_${compass_gait_walker}.csv
python Plotting/PlotAlgorithmSets.py Data/ExpanderContractor_ellipsoid_log_${expander_contractor}.csv

# Make Invariant Set Plots (Fig 2c, 2d)
python Plotting/PlotInvariantSet.py Data/CompassGaitWalker_ellipsoid_log_${compass_gait_walker}.csv
python Plotting/PlotInvariantSet.py Data/ExpanderContractor_ellipsoid_log_${expander_contractor}.csv
