import numpy as np

__author__ = "Joshua Mangelson, Joshua Broekhuijsen(translation from MATLAB), and Ryan Eustice(MATLAB)"
__copyright__ = "Copyright 2022 - Joshua Mangelson, Joshua Broekhuijsen, and Ryan Eustice"
__license__ = "MIT License"
__maintainer__ = "Joshua Mangelson"


def detectTrees(laserScan):
    """
        This function takes the laser scan (361 measurements, counterclockwise from 0 to pi radians)
        and identifies trees (landmarks) according to some user-controllable parameters.
        The original MATLAB was designed to run efficiently rather than accurately, so I've
        done my best to follow the original intent of the algorithm where necessary.  Results
        from this should almost always line up with the MATLAB version, but might be a little 
        more restricted and accurate.  Worth benchmarking to see how they perform relatively.

        Note that the original MATLAB accepted doubles (in meters) and used int16 for internal computation.
        It's no longer 1999, so I'm using cm (32-or-64-bit ints) for computation to avoid floats.

        Parameters:
            1x361 int laser scan in cm

        Returns:
            nx3 array of features, [dist (m), angle (radians), diameter (m)]
    """
    
    
    
    # --------------------------------
    # PARAMETERS
    # --------------------------------
    maxLaserRange = 7500                        # (cm), This represents the maximum range we'll 'trust' from the laser.  The laser reports up to 82m, we're truncating a bit early.
    minLaserRange = 100                         # (cm), This represents the minimum range for a landmark to be detected.  Closer than this might be a faulty reading.
    trustedAngle = 2.5*np.pi/180                # (radians), Landmarks closer to the edge of our range than this will be discarded as untrustworthy. Original MATLAB was 5*pi/306, which was probably a typo.
    radialDiscontinuityThreshold = 150          # (cm), When looking at laser ranges, an abrupt change of more than this indicates a potential landmark boundary
    angularDiscontinuityThreshold = 5*np.pi/180 # (radians), After max-range measurements are discarded, this is used to indicate potential landmark boundaries
    minimumLandmarkDistance = 300               # (cm), The minimum required distance between landmarks (or else we discard something) to avoid ambiguity
    maximumLandmarkDiameter = 100               # (cm), the maximum diameter of a landmark.  Beyond this, it's probably not a tree, and we don't want to use it.
    minimumLandmarkAngularDist = 1*np.pi/180    # (radians), The minimum required angular distance between landmarks (or else we discard something) to avoid ambiguity
    # --------------------------------
    measurementAngles = np.arange(361)*np.pi/360 # these are the angles at which the measurements from the laser occur, relative to the agent

    # We'll begin by selecting ONLY the measurements from the laser that are less than our maximum threshold.
    nonMaxIndices = np.flatnonzero(np.asarray(laserScan < maxLaserRange))
    if (len(nonMaxIndices) == 0): # and short circuit for speed if we have none
        return np.array([])

    # Select those items out so we can work with them.
    nonMaxMeasurements = laserScan[nonMaxIndices]
    nonMaxAngles = measurementAngles[nonMaxIndices]
    
    # Next we'll search and find the indices in the data at which sharp discontinuities in angle or range appear
    discontinuityIndices = np.flatnonzero(np.asarray((np.abs(np.diff(nonMaxMeasurements)) > radialDiscontinuityThreshold) | (np.diff(nonMaxAngles) > angularDiscontinuityThreshold)))
    
    # The "diff" function reduced the size of our indices by one, so we need to restore that.  We'll want one array holding the "lower side" indices, and one holding the "upper side"
    # ldi = lower dicontinuity indices, udi = upper discontinuity indices, abbreviated to avoid annoying typing.
    ldi = np.array([0, *(discontinuityIndices + 1)]) # the +1 is to solve the off-by-one that happened from "diff"
    udi = np.array([*discontinuityIndices, len(nonMaxIndices)-1])

    # With these indices, we can get the actual measurements and angles at those indices for comparisons.
    # ldr = lower discontinuity radii, udr = upper discontinuity radii, lda = lower discontinuity angles, uda = upper discontinuity angles
    ldr = nonMaxMeasurements[ldi]; udr = nonMaxMeasurements[udi]
    lda = nonMaxAngles[ldi]; uda = nonMaxAngles[udi]
    
    # Using the values extracted above, we can find the x and y coordinates of the discontinuities relative to the agent for future calculations
    # lx = lower x, rx = upper x, ly = lower y, ry = upper y
    whichType = float # can switch this to "int" for slightly faster computation.  Not sure if it matters much.
    lx = (ldr*np.cos(lda)).astype(whichType); ux = (udr*np.cos(uda)).astype(whichType)
    ly = (ldr*np.sin(lda)).astype(whichType); uy = (udr*np.sin(uda)).astype(whichType)
    
    invalidFlag = np.zeros_like(ldr).astype(int) # If this flag is set, the landmark should be considered invalid (ambiguous or otherwise not meeting parameters) and not returned
    
    sqDist = minimumLandmarkDistance**2 # We'll compare squared distances rather than actual distances to avoid using sqrt for speed

    # This next step is... complicated.  The MATLAB only compared 3 levels, but we'll compare all.  Each row contains a list of the squared
    # x or y distance from the landmark at index (row index) to the landmark at index (col index + 1).
    # Note that we're also taking only the upper triangle.  This is because we start measuring things weirdly
    # (like the width of a landmark itself, or the distance between unhelpful edges) in the lower region.  We'll 0 them for faster computation
    # Since we're not going to use them anyway
    xDiff = np.triu(np.subtract.outer(ux[:-1], lx[1:]))**2
    yDiff = np.triu(np.subtract.outer(uy[:-1], ly[1:]))**2

    # If we took the sqrt of each element, we'd have euclidian distances.  Square roots are expensive, though,
    # so we don't want to do that.  Intead we'll compare squared distances, since they're monotonic.
    sqEucDiff = xDiff + yDiff

    # We've got the sqEucDiff, but the 0s we put in as masks will mess up computation.  Accordingly, we'll add in some
    # non-zero (and greater than the check we're going to run) values to avoid this.
    sqEucDiff += np.tril(np.full_like(sqEucDiff, sqDist + 1), -1)

    # We're checking each row to see if any of the column values (distances) are less than our threshold (and therefore ambiguous)
    invalidFlag[np.flatnonzero(np.asarray(np.min(sqEucDiff, axis=1) < sqDist))] = 1 # Set the flags anywhere the distance is not sufficient.  We'll remove the lower-indexed landmark.

    # Now we'll check to see if any of the angular differences are too small.  We don't have to repeat the same trick as we did for the euclidian distance,
    # since the angles between consecutive landmarks are monotonically increasing; we only need to check one level.
    angleTooSmallIndices = np.flatnonzero(np.asarray(lda[1:] - uda[:-1] < minimumLandmarkAngularDist))

    # In the case where they're too close angularly, we'll find which one is in the back and remove it, since it's obscured.
    inBack = ldr[angleTooSmallIndices + 1] > udr[angleTooSmallIndices]

    # We now update our array with the ones to remove
    angleTooSmallIndices += inBack

    # And flag them
    invalidFlag[angleTooSmallIndices] = 1
    
    # Now we need to extract the data for the non-flagged indices.
    unflaggedIndices = np.flatnonzero(np.asarray(invalidFlag != 1))

    # lfr = lower filtered radii, ufr = upper filtered radii, lfa = lower filtered angle, ufau = upper filtered angle
    # lfx = lower filtered x, ufx = upper filtered x, lfy = lower filtered y, ufy = upper filtered y
    lfr = ldr[unflaggedIndices]; ufr = udr[unflaggedIndices]
    lfa = lda[unflaggedIndices]; ufa = uda[unflaggedIndices]
    lfx = lx[unflaggedIndices]; ufx = ux[unflaggedIndices]
    lfy = ly[unflaggedIndices]; ufy = uy[unflaggedIndices]

    #cil = carry indices lower, ciu = carry indices upper
    # We need to hold on to these indices (relative to unfiltered data) for computation at the end
    cil = ldi[unflaggedIndices]
    ciu = udi[unflaggedIndices]
    
    # We now need to check to make sure the diameter is smaller than our maximum threshold (to find things that are sensible for trees)
    sqDiam = (lfx - ufx)**2 + (lfy - ufy)**2
    treeIndices = np.flatnonzero(np.asarray(sqDiam < maximumLandmarkDiameter**2))

    # We'll just apply that filter really quickly  
    lfr = lfr[treeIndices]; ufr = ufr[treeIndices]; lfa = lfa[treeIndices]; ufa = ufa[treeIndices]
    cil = cil[treeIndices]
    ciu = ciu[treeIndices]

    # Out next filter is that the minimum distance for the landmark (from the agent) must be 1m - we'll just check the lower edge for speed
    # Additionally, if any part of the landmark is in the edges of our ability to sense (trustedAngle), we'll discard it.
    indexFilter = np.flatnonzero(np.asarray((lfr > minLaserRange) & (lfa > trustedAngle) & (ufa < (np.pi - trustedAngle))))

    # Aaaand apply....
    lfr = lfr[indexFilter]; ufr = ufr[indexFilter]; lfa = lfa[indexFilter]; ufa = ufa[indexFilter]
    cil = cil[indexFilter]
    ciu = ciu[indexFilter]
    
    # Now we need to find the diameter of the landmark.  Rather than using trig functions, we'll use the arc length
    # Arc length is faster and preeeeeetty much as accurate on the scale we're dealing with.
    landmarkDiameter = (ufa + np.pi/360 - lfa)*((lfr + ufr)/2)

    # Time for - you guessed it - ANOTHER filter.  This time we'll be checking something a bit more complicated.
    # Calculate the dist/radius to the left side of the landmark (from the agent), and the dist to the right side
    # Make sure that distance is less than 1/3 the diameter of the landmark in question.
    # Honestly, I'm not sure I totally understand this filter, but it was in the original MATLAB.  My guess is that it's
    # Checking that the landmark is vaguely circular (as you would expect for a tree), but accounting for sensor slop
    indexFilter = np.flatnonzero(np.asarray(np.abs(ufr - lfr) < landmarkDiameter/3))

    # Aaaaaand apply....
    lfr = lfr[indexFilter]; ufr = ufr[indexFilter]; lfa = lfa[indexFilter]; ufa = ufa[indexFilter]; landmarkDiameter = landmarkDiameter[indexFilter]
    cil = cil[indexFilter]
    ciu = ciu[indexFilter]

    # We're almost done, for real.  We need to find the sensor measurement that points most closely at the center of each landmark.
    # We'll use that to caluclate the average dist from the sensor to the start (closest point) of the landmark
    centerIndices = (cil + ciu)/2
    #cil = center indices lower, ciu = center indices upper
    cil = np.floor(centerIndices).astype(int)
    ciu = np.ceil(centerIndices).astype(int)
    averageRadius = (nonMaxMeasurements[cil] + nonMaxMeasurements[ciu])/2

    # Finally, we'll construct our return
    outStructure = np.column_stack([(averageRadius + landmarkDiameter/2)/100, (lfa + ufa)/2, landmarkDiameter/100])

    return outStructure
    
