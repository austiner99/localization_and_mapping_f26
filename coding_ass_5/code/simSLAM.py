import numpy as np
from generator import generate
import time
from fieldSettings import field
import helpers
import matplotlib

# this might be necessary on MacOS for live plotting, dunno what it'll do on other systems
matplotlib.use("Qt5Agg")

import matplotlib.pyplot as plt
dpi = 147
plt.rcParams["figure.figsize"] = (4 * dpi/80, 4 * dpi/80) # size of plots, set to 8x8 inches (default 80ppi, so might be be the same on a given display)

# Can change this to true if you need some debugs in your code, I did it on specific timesteps where I had issues
helpers.printDebug = False 
from helpers import debugPrint

def run(numSteps, dataAssociation, updateMethod, pauseLen, makeGif):

    # Video/GIF generation
    printNumLength = len(str(numSteps)) # Used to name gif images properly
    if (makeGif): # If we're going to be making a gif/video of the output,
        from PIL import Image # we need PIL to do so

    # initialize and settings
    initialStateMean = np.array([180, 50, 0]) # initial state, [x, y, theta]
    maxObs = 2 # The maximum number of features that will be observed at any timestep
    alphas = np.array([0.05, 0.001, 0.05, 0.01])**2 # These have to do with the error in our motion controls
    betas = np.array([10, 10*np.pi/180]) # Error in observations
    R = np.diag(betas**2)
    deltaT = 0.1 # Time between each timestep, must be < 1.0s
    
    # NOTE: The below will pull a previously-generated dataset if one exists for the format you've selected.  If this is undesireable, call with "forceNew=True"
    data = generate(initialStateMean, numSteps, alphas, betas, deltaT, maxObs, forceNew=True) # there are other options in this function def for what landmarks are observed, we won't mess with them

    # For ease of the algorithm below, we'll just define our "dataAssociation" function now by importing
    if (dataAssociation == "known"): # Known is only for the simulator
        from da_known import associateData
    elif (dataAssociation == "nn"): # Nearest neighbor here is in terms of mahalanobis distance, not euclidean
        from da_nn import associateData
    elif (dataAssociation == "nndg"): # Nearest neighbor double gate adds a "no mans land" of ingored observations
        from da_nndg import associateData
    elif (dataAssociation == "jcbb"): # joint compatibility considers all the measurements at once
        from da_jcbb import associateData
    else:
        raise Exception('Unknown data association method.')

    #===================================================
    # Initialize State
    #===================================================
    realRobot = initialStateMean.copy() # This will be the variable we propagate
    realCov = 1e-03*np.eye(3) # start covariance low but non-zero
    landmarkSignatures = [] # This is to track the signature (markerID for known association) of landmarks.  Only used for da_known.
    # Track Landmark Associations for plotting later
    landmarkAssociationHistory = np.zeros((25, 26))

    muHist = np.zeros((numSteps, 2)) # Tracking the position of the robot over time for plotting

    plt.ion() # Interacting plotting so we can SEE it doing its thing
    for t in range(numSteps): # We'll run our algorithm for each step of data
        print("Running step ", t, " currently have ", int((len(realRobot)-3)/2), " landmarks ") # Not necessary, just nice to see.
            
        #=================================================
        # Data available to your filter at this time step
        #=================================================
        u = data[t, 0:3] # [d_rot_1, d_trans, d_rot_2]
        temp = data[t, 9:] # [id or -1 if no obs, noisy dist, noisy theta, noise-free dist, noise-free theta, repeat....]
        z = [] # We need to extract measurements.  Simulated data has a "-1" for id if there isn't an observation in that slot for this timestep.
        for i in range(int(len(temp)/5)): # So we need to check each
            if (temp[i*5] != -1): # and see if it's real or not
                z.append(temp[i*5:i*5+3]) # Then grab the first 3 values, since they have noise - last 2 are noise-free, not allowed in our simulator.
        z = np.array(z) # [[id, dist, theta], [id, dist theta], etc]
        order = np.array([1, 2, 0]) # But we want order "dist, theta, id" to match VP.
        z = ((z.T)[order]).T # Reorder, and now our measurements are ready to go.

        #=================================================
        # Update based on Robot Motion
        #=================================================

        # TODO: Setup to Handle Noisy Controls
        M = np.eye(3) # Initialized with Identity - to make things run - Need to fix this to be correct
        
        # Call your predict function
        # TODO: Implement the predict function defined below
        realRobot_bar, realCov_bar = predict(realRobot, realCov, u, M)

        #=================================================
        # Update based on Measurements
        #=================================================
        
        
        if (len(z) > 0): # If we HAVE any measurements...
            # Call the associateData function to determine which measurements correspond to which landmarks
            # TODO: Implement each of the associateData functions in the da_----.py files
            association, H, innovation = associateData(z, R, realRobot_bar, realCov_bar, landmarkSignatures) 
            
            # Save a history of associations so you can plot them later
            for index in range(len(association)):
                trueId = int(z[index][2])
                associatedTo = int(association[index] + 1)
                landmarkAssociationHistory[trueId][associatedTo] = landmarkAssociationHistory[trueId][associatedTo] + 1

            #########################################################
            # TODO: Here you should call the update and augmentState functions
            # Commented out until implemented
            
            # Update your state for landmarks already observed
            #realRobot, realCov = update(realRobot_bar, realCov_bar, association, H, R, innovation, updateMethod) 

            # Augment our state with new landmarks that were not associated
            #realRobot, realCov, landmarkSignatures = augmentState(association, z, realRobot, realCov, R, landmarkSignatures)

            # Comment this out after you implement and un comment the above two function calls
            realRobot = realRobot_bar
            realCov = realCov_bar
            
        #=================================================
        #TODO: plot and evaluate filter results here
        #=================================================
        muHist[t] = realRobot[:2] # Track the position of the robot over time
    
        plotsim(data, t, initialStateMean, realRobot, realCov, muHist); # Plot the state as it is after the timestep
        plt.legend()        
        plt.gcf().canvas.draw() # Tell the canvas to draw, interactive mode is weird
        if pauseLen > 0: # If we've got a pauselen, let's take a break so it doesn't blur past
            time.sleep(pauseLen)

        # Save GIV Data
        if (makeGif): # If we're saving to make a video, let's put the current frame into a saved image for later processing.
            imgData = np.frombuffer(plt.gcf().canvas.tostring_rgb(), dtype=np.uint8) # Extra image data from the plot
            w, h = plt.gcf().canvas.get_width_height() # Determine the dimensions
            mod = np.sqrt(imgData.shape[0]/(3*w*h)) # multi-sampling of pixels on high-res displays does weird things, account for it.
            im = imgData.reshape((int(h*mod), int(w*mod), -1)) # Create our image array in the right shape
            Image.fromarray(im).save("outputGif/sim_" + str(t).zfill(printNumLength) + ".png") # And pass it to PIL to save it.

        # Make sure the canvas is ready to go for the next step
        plt.gcf().canvas.flush_events() 

        
        
    # END of Loop    
    plt.ioff() # Once we've done all time steps, turn off interactive mode
    print("Finished execution") # Nice to know when we're done since sometimes the data moves slowly
    np.savez("nnAssociation.npz", associationHistory=landmarkAssociationHistory)

    # GIF Plotting
    if (makeGif):
        import glob # Need this to load photos (better than holding them in RAM forever)
        # Create the frames
        frames = [] # This holds each image
        imgs = glob.glob("outputGif/sim_*.png") # load 'em in
        imgs.sort() # Make sure they're in the right order
        for i in imgs: # For each one, we'll open and append
            new_frame = Image.open(i)
            frames.append(new_frame)
        
        # Save into a GIF file that loops forever
        frames[0].save('outputGif/output.gif', format='GIF',
                       append_images=frames[1:],
                       save_all=True,
                       duration=len(imgs)*0.05, loop=1)
    plt.show() # And just show the last image
    
#==========================================================================
# Pretty drawing stuff
#==========================================================================
def plotsim(data, t, initialStateMean, mu, Sigma, muHist):

    noiseFreePathColor = '#00FF00'
    noisyPathColor = '#0000FF'
    estimatedPathColor = '#FFBB00'

    noiseFreeBearingColor = '#00FFFF'
    observedBearingColor = '#FF0000'
        
    #=================================================
    # data *not* available to your filter, i.e., known
    # only by the simulator, useful for making error plots
    #=================================================
    # actual position (i.e., ground truth)
    x, y, theta = data[t, 3:6]
    
    # real observation
    z = data[t, 9:] # [id or -1 if no obs, noisy dist, noisy theta, noise-free dist, noise-free theta, repeat....]
    obs = [] # We need to gather together the observations that are non-null
    for i in range(int(len(z)/5)):
        if (not (z[i*5] == -1)):
            obs.append(z[i*5:i*5+5])
    obs = np.array(obs)

    #################################################
    # Graphics
    #################################################
    plt.clf() # clear the frame.
    helpers.plotField(obs[:,0]) # Plot the field with the observed landmarks highlighted

    # draw actual path and path that would result if there was no noise in
    # executing the motion command
    plt.plot(np.array([initialStateMean[0], *data[:t, 6]]), np.array([initialStateMean[1], *data[:t, 7]]), color=noiseFreePathColor, label='Noise Free Path')
    plt.plot(data[t, 6], data[t, 7], '*', color=noiseFreePathColor)

    # draw the path that has resulted from the movement with noise
    plt.plot(np.array([initialStateMean[0], *data[:t, 3]]), np.array([initialStateMean[1], *data[:t, 4]]), color=noisyPathColor, label='True Noisy Path')
    helpers.plotRobot(data[t, 3:6], "black", "#00FFFF40")

    # draw the path the estimated robot followed
    plt.plot(np.array([initialStateMean[0], *muHist[:t, 0]]), np.array([initialStateMean[1], *muHist[:t, 1]]), color=estimatedPathColor, label='EKF SLAM Est')
    plt.plot([mu[0]], [mu[1]], '*', color=estimatedPathColor)
    helpers.plotCov2D(mu[:2], Sigma[:2, :2], color=estimatedPathColor, nSigma=3)

    for observation in obs:
        # indicate observed angle relative to actual position
        plt.plot(np.array([x, x+np.cos(theta + observation[2])*observation[1]]), np.array([y, y+np.sin(theta + observation[2])*observation[1]]), color=observedBearingColor)

        # indicate ideal noise-free angle relative to actual position
        plt.plot(np.array([x, x+np.cos(theta + observation[4])*observation[3]]), np.array([y, y+np.sin(theta + observation[4])*observation[3]]), color=noiseFreeBearingColor)
    
    for i in range(int((len(mu) - 3)/2)):
        # We'll also plot a covariance ellipse for each landmark
        helpers.plotCov2D(mu[3+i*2:3+i*2+2], Sigma[3+i*2:3+i*2+2, 3+i*2:3+i*2+2], nSigma=3)

##################################################
# TODO: Implement the Motion Prediction Equations
##################################################
def predict(mu, Sigma, u, M):
    # Propogate Jacobian

    # Find Jacobians

    # Update Mean and Covariance (Handle both Robot State and Landmarks)
    mu_bar = mu # initialized to make things run, needs to be fixed
    Sigma_bar = Sigma # initialized to make things run, needs to be fixed
    
    return mu_bar, Sigma_bar # And give them back to the calling function

###############################################
# TODO: Implement Measurement Update Equations
###############################################
def update(mu_bar, Sigma_bar, association, H, Q, innovation, updateMethod):
    ind = np.flatnonzero(np.asarray(association > -1)) # -1 is used as a keyword for "new landmark," -2 is used for "ignore"
    if (len(ind) == 0): # If we don't have any, short-circuit return
        return mu_bar, Sigma_bar

    if (updateMethod == "seq"): 
        for i in ind: 
            # TODO: Finish This to update incrementally for each measurement
            pass # To enable running until implemented
    elif (updateMethod == "batch"): 
        # TODO: Finish This to update all measurements of a time step at once
        pass # To enable running until implemented        
    else:
        raise Exception("Unknown update method, '" + updateMethod + "' - it must be 'seq' or 'batch'")
    
    return mu_bar, Sigma_bar

##################################
# TODO: Implement Augment State
##################################
def augmentState(association, measurements, mu, Sigma, Q, landmarkSignatures):
    indices = np.flatnonzero(np.asarray(association == -1)) # If our data association returned a "-1" for a measurement, it's a landmark to add to the state
    measurements = measurements[indices] # We want to filter out only the measurements we cared about
    x, y, theta = mu[:3] # We'll need the robot state
    Sigma_rr = Sigma[:3, :3] # And the main robot covariance, we won't change
    
    # For each measurement of a new landmark update your state
    for z in measurements:
        # Extract info for the measurement
        dist, bearing, sig = z 
        # Update the signatures so we know what landmark index goes to what signature.  Only used for da_known
        landmarkSignatures = np.array([*landmarkSignatures, sig]) 

        # Update both the mean (mu_l) and covariance (Sigma_lr, Sigma_rl, Sigma_Ll, Sigma_lL, Sigma_ll) for the new landmark.
    
    return mu, Sigma, landmarkSignatures
