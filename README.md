# Ex0-GNSSRawMeasurments

In this assignment, we were asked to take the logs from the GnssLogger app and get the location of where the logs were taken from using the log's raw data. 

To read more about the app look <a href="https://developer.android.com/develop/sensors-and-location/sensors/gnss">here</a>.

To download the app go to Google Play Store or Apple Store.

To run our code do the following:

gitclone "" fill in later
<br> run the following to make sure you have all the libraries needed
<br> pip install -r /path/to/requirements.txt 

-----------------------------------------------
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

-----------------------------
### About our enterative algorithm

------------
### How to view the kml file
One way i by using <a href="https://www.google.com/maps/d/">mymaps</a> whish is part of google maps.
<br>Click on Create a new map.
<br>Click on import and browse, to choose the load the file.


Here are images of the locations from some of the log files that we used.
<br> add pics
 
