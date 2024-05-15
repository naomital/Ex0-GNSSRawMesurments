# Autonomous Robotics -  Ex0: GNSS Raw Measurements

This project implements a basic positioning algorithm based on Root Mean Square (RMS) of weighted pseudo-ranges for a given GNSS log file.

### Purpose

This Python code solves an assignment on the fundamentals of GNSS, including:

- Understanding pseudo-ranges
- Parsing GNSS raw measurement data into a CSV format
- Implementing a least-squares positioning algorithm
- Converting positions from ECEF coordinates to latitude, longitude, and altitude
- Creating KML and CSV output files to visualize the path and calculated positions

## How to Use


### 2. Install dependencies:
This project requires the following Python libraries:

﻿numpy==1.23.5
pandas==2.0.3
pyproj==3.6.1
simplekml==1.3.6

> **NOTE**: check **requiremets.txt**

```
### 3. Select GNSS log file

the log file need to be recorded on GnssLogger app from Google

To read more about the app look <a href="https://developer.android.com/develop/sensors-and-location/sensors/gnss">here</a>.

To download the app go to <a href="https://play.google.com/store/apps/details?id=com.google.android.apps.location.gps.gnsslogger">Google Play Store</a> or Apple Store.

### 4.Run the script:
The script is designed to work on desired log file 
You can insert in the script in main() to use a file path.

Navigate to the root directory of the project and run:

```sh
python rawMeasurementToGPS.py
```
### About our code

The majority of our code was taken from this <a href="https://www.johnsonmitchelld.com/2021/03/14/least-squares-gps.html">website</a> that we were refrenced to use 
and its corisponding <a href="https://www.johnsonmitchelld.com/2021/03/14/least-squares-gps.html">github</a> account.
<br>We used the code mentioned above to procces the raw data into a dataframe and get information about the satalights and their locations.
<br>Than we exported that data to a csv file, so that our code would work if we are also given this csv file.
<br>Next, we read the csv file and wrote our own code to enteratively find the location from where the logs were taken.
<br>After that we retured two files:
<br>The first is a csv with important raw data about the satalites and the location of were the logs were taken.
<br>The second is a kml file with the one location per second.

For our code to run in the secound part we need to get a csv file with the following columns:
<br> sat_name, UnixTime, GpsTimeNanos, tTxSeconds, Cn0DbHz, PrM, delT_sv, Epoch, x_sat, y_sat, z_sat

The final csv has all of the above columns and pos_x, pos_y, pos_z, lat, lon, alt

### About our iterative algorithm

This function implements a basic iterative algorithm for least-squares direction finding. It takes several inputs:

- `xs`: Positions of reference points (likely satellites).
- `measured_pseudorange`: Measured distances to the reference points.
- `x0`: Initial guess for the target position.
- `b0`: Initial guess for a bias term.

The algorithm works by:

1. Calculating the initial error based on the difference between measured and predicted pseudoranges.
2. It iterates 200 times (or until convergence) trying different small adjustments (dx) to the target position in each dimension (x, y, z).
3. For each adjustment, it calculates a new error based on the updated position.
4. It keeps the adjustment that leads to the lowest error and updates the target position and bias term accordingly.
5. The process stops when the adjustments become very small (indicating convergence) or after a fixed number of iterations.

**Essentially, it refines the target position and bias term iteratively to minimize the difference between measured and predicted distances to reference points.**


> **NOTE**: This is a simplified explanation. The actual implementation uses techniques like line search and stopping criteria for better performance.





## Output

The script will generate two output files in the same directory as the input file:

- \<filename>_answer.csv: This CSV file contains the original GNSS measurements with additional columns for the calculated X, Y, Z positions, latitude, longitude, and altitude.


- \<filename>.kml: This KML file is a visualization of the calculated positions as a path. You can open this file with Google Earth or any other KML reader

### How to view the kml file
One way is by using Google <a href="https://www.google.com/maps/d/">MyMaps</a> whish is part of google maps.
<br>Click on Create a new map.
<br>Click on import and browse, to choose the load the file.

### Output exaples:
<img src="images/צילום מסך 2024-05-15 204840.png" width="325"> <img src="images/צילום מסך 2024-05-15 204926.png" width="325">
<img src="images/צילום מסך 2024-05-15 205509.png" width="305"> <img src="images/צילום מסך 2024-05-15 205710.png" width="375">
<img src="images/צילום מסך 2024-05-15 204957.png" width="485"> <img src="images/צילום מסך 2024-05-15 205429.png" width="132">

## Further Considerations

This is a basic implementation and can be extended to include features like:

- Filtering and weighting of pseudo-ranges based on signal quality
- Integration with real-time GNSS data streams
- Implementation of more advanced positioning algorithms

We hope this helps! Feel free to reach out if you have any questions.

