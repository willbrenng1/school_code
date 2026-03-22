README - CSE6242 Project (Group 094)


====DESCRIPTION====

Group 094’s “Optimizing Traffic Cameras in D.C. to Reduce Crashes is an interactive web application built via Python and Plotly-Dash. Crash data across Washington, D.C. is analyzed to determine high-risk areas where new speed cameras would most effectively reduce speeding-related accidents and is designed for city planners to explore budget-constrained safety improvements visually.

The process for this system:
	-Loads crash data and traffic camera location datasets from DC Open Data
	-Clusters crash points using DBSCAN to find crash hotspots
	-Calculates crash cluster centroids and their distance from existing cameras
	-Uses Pulp to recommend optimal new camera placements given a user-defined budget
	-Displays results using Plotly Mapbox with options to view existing cameras, clusters, and recommended locations


====INSTALLATION====

Ensure Python 3.8+ is installed
Place the following files in the same directory:
	-094_pipeline.py
	-Automated_Traffic_Enforcement_Table.csv
	-Crashes_in_DC.csv
	-Automated_Traffic_Enforcement_Table.geojson
Install required libraries:
- Run:
	 pip install -r requirements.txt


====EXECUTION====
-Open a terminal in your project directory
-Run:
	python 094_pipeline.py
-Upon startup, open your browser and navigate to:
	http://127.0.0.1:8070
-In the UI of the app:
	-Enter a budget a select “Update”
	-Toggle ‘Show Cameras’ (on by default) to remove existing cameras
	-Toggle ’Show Full Cluster’ to see the individual crashes that make up the current recommendation centroid
	-View new recommended camera locations on the map (number of recommendations based on budget)
	-Select rows in the results table (far left circular button) to highlight that particular point on the map
	-Use ‘Download CSV’ to export the recommendations

====WALKTHROUGH DEMO====
https://youtu.be/k2oPhnjasDo







