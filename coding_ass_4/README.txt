# Lab 4 - EKF and UKF Localization

## Code

The provided files are outlined as follows:
- run.py - This is the main file that will run the simulation. ("./run.py -h" for arguments)
- pfUpdate.py - Here you should add your code to implement your PF. (Not needed in Lab 4).
- ekfUpdate.py - Here you should add your code to implement your EKF. (Lab 4)
- ukfUpdate.py - Here you should add your code to implement your UKF. (Lab 4)
- helpers.py - This file has a variety of helper functions that you may use if you so choose while implementing your filters.
- fieldSettings.py - This file generates the field. You do not need to modify this file. 
- generator.py - This file generates a noisy trajectory/dataset when running the simulation without a passed in dataset. You do not need to modify this file.
- testBench.py - This file is extra for you to use while testing if you so choose.

There are various options that can be set when executing run.py
Run ```./run.py -h``` to see all the options listed.

For example, running the following command will run the simulator with the EKF filter (as opposed to all filters), taking data from the data/task2.npz file, and with a frame rate of 0.01 seconds for visualization:
```
python code/run.py -d data/data.npz -a ekf -w 0.01
```

## Data

The data to be used for your final results and writeup is stored in data/task2.npz

You may use the simulator to generate random data as much as you would like when testing, but the primary plots and videos you submit with your writeup should use the data in task2.npz.

## Pointers

Here are some pointers to help you along:
- The alpha and beta parameters at the top of the run function set the noise levels that the filters use for tracking motion and sensor noise. These default values match the dataset in task2.npz. They are also passed into the generate() function that generates noisy data. For the filter to work optimally, it needs to have noise parameters that match the data, so don't lose these values. However, in debugging it can sometimes be helpful to temporarily decrease noise to zero for either motion or measurements noise or both and test to make sure your filter works correctly.
- It can also be helpful to comment out the update or correction steps and just focus on one at a time to make sure things work as expected.


## Lab Instructions

The instructions for the lab are contained in ekf-ukf-landmark-localization-lab.pdf

