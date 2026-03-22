# %%
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from geopy.distance import geodesic
import geopandas as gpd
import pulp
import folium
import osmnx as ox
import folium
import streamlit as st
from streamlit_folium import st_folium


# %% [markdown]
# import and cleaning

# %%
cam_locations = pd.read_csv(r"Automated_Traffic_Enforcement_Table.csv")

crash_data = pd.read_csv(r"Crashes_in_DC-2.csv")

# %%
crash_data = crash_data[crash_data["SPEEDING_INVOLVED"] >= 1]

# %%
crash_data["REPORTDATE"] = pd.to_datetime(crash_data["REPORTDATE"])
crash_data_filtered = crash_data[
    crash_data["REPORTDATE"].between("2010-01-01", "2025-10-17", inclusive="both")
]
crash_data_filtered = crash_data_filtered.rename(
    {"LATITUDE": "latitude", "LONGITUDE": "longitude"}, axis=1
)

# %%
cam_locations["START_DATE"] = pd.to_datetime(cam_locations["START_DATE"])
activecam_locations = cam_locations[
    cam_locations["ACTIVE_STATUS"] != "Retired"
].reset_index()

active_speed_cam_locations = activecam_locations[
    activecam_locations["ENFORCEMENT_TYPE"] == "Speed"
]
cam_data_w_coordinates = active_speed_cam_locations[
    active_speed_cam_locations["CAMERA_LATITUDE"].notna()
]

# %%
crash_data_filtered = crash_data_filtered.rename(
    {"LATITUDE": "latitude", "LONGITUDE": "longitude"}, axis=1
)

# %% [markdown]
# crash clustering


# %%
def cluster_crashes(df, min_sample, eps_meters):

    coords = df[["latitude", "longitude"]].to_numpy()

    eps_km = eps_meters / 1000
    kms_per_radian = 6371.0088
    eps = eps_km / kms_per_radian

    db = DBSCAN(eps=eps, min_samples=min_sample, metric="haversine").fit(
        np.radians(coords)
    )

    df["cluster_id"] = db.labels_

    return df


# %%
# used 80m for ~standard city block size
crash_clusters = cluster_crashes(crash_data_filtered, 10, eps_meters=80)

# %%
# crash density per cluster
cluster_sizes = (
    crash_clusters.groupby("cluster_id", as_index=False)
    .size()
    .rename({"size": "crash_count"}, axis=1)
)

# %%
crash_geo_df = gpd.GeoDataFrame(
    crash_clusters,
    geometry=gpd.points_from_xy(
        crash_clusters["longitude"], crash_clusters["latitude"]
    ),
)

# %%
# getting central point for each cluster
cluster_centroids = (
    crash_geo_df[crash_geo_df["cluster_id"] != -1].dissolve(by="cluster_id").centroid
)

crash_cluster_centroids = cluster_centroids.reset_index(drop=False)
crash_cluster_centroids.columns = ["cluster_id", "crash_centroid_coords"]


# %%
crash_cluster_centroids["cluster_id"].duplicated().sum()

# %%

crash_geo_df = gpd.GeoDataFrame(
    crash_cluster_centroids, geometry="crash_centroid_coords", crs="EPSG:4326"
)

# %%
# getting geodf for cam locations
cam_locatios_geo = gpd.GeoDataFrame(
    cam_data_w_coordinates,
    geometry=gpd.points_from_xy(
        cam_data_w_coordinates["CAMERA_LONGITUDE"],
        cam_data_w_coordinates["CAMERA_LATITUDE"],
    ),
    crs="EPSG:4326",
)[["ENFORCEMENT_SPACE_CODE", "geometry"]]

# %%
cam_locations_proj = cam_locatios_geo.to_crs(epsg=26918)
crash_locations_proj = crash_geo_df.to_crs(epsg=26918)

# %%
# getting nearest cam to crash clusters
nearest_cameras = gpd.sjoin_nearest(
    crash_locations_proj, cam_locations_proj, distance_col="distance_m"
)


# %%
nearest_cameras = nearest_cameras.drop_duplicates(subset="cluster_id")

# %%
nearest_cameras = nearest_cameras.sort_values(
    by="distance_m", ascending=True
).drop_duplicates(subset="cluster_id", keep="first")

# %%
# merging and renaming
neartest_cam_to_crash_clusters = nearest_cameras.merge(
    cluster_sizes, on="cluster_id", how="left"
)
neartest_cam_to_crash_clusters = neartest_cam_to_crash_clusters.rename(
    {"crash_centroid_coords": "crash_centroid_coords_projection"}, axis=1
)
neartest_cam_to_crash_clusters = neartest_cam_to_crash_clusters.merge(
    cam_locatios_geo, on="ENFORCEMENT_SPACE_CODE", how="left"
).rename({"geometry": "camera_coordinates"}, axis=1)
neartest_cam_to_crash_clusters = neartest_cam_to_crash_clusters.merge(
    crash_cluster_centroids, on="cluster_id", how="left"
)
neartest_cam_to_crash_clusters = neartest_cam_to_crash_clusters.rename(
    {
        "crash_centroid_coords": "crash_centroid_coords_coordinates",
        "ENFORCEMENT_SPACE_CODE": "cam_name",
        "distance_m": "nearst_camera_distance",
    },
    axis=1,
).drop("index_right", axis=1)

# %% [markdown]
# Recomendation Model

# %%
neartest_cam_to_crash_clusters["camera_coordinates_longitude"] = (
    neartest_cam_to_crash_clusters["camera_coordinates"].x
)
neartest_cam_to_crash_clusters["camera_coordinates_latitude"] = (
    neartest_cam_to_crash_clusters["camera_coordinates"].y
)

neartest_cam_to_crash_clusters["crash_centroid_longitude"] = (
    neartest_cam_to_crash_clusters["crash_centroid_coords_coordinates"].x
)
neartest_cam_to_crash_clusters["crash_centroid_latitude"] = (
    neartest_cam_to_crash_clusters["crash_centroid_coords_coordinates"].y
)

# %%
# filtering for proximity to current camera locations
neartest_cam_to_crash_clusters_over_eighty_meters = neartest_cam_to_crash_clusters[
    neartest_cam_to_crash_clusters["nearst_camera_distance"] > 80
]

# %%
neartest_cam_to_crash_clusters_over_eighty_meters = (
    neartest_cam_to_crash_clusters_over_eighty_meters.set_index("cluster_id")
)

# %%
# calculating distance between crash clusters, running slow room to improve
crash_sites = neartest_cam_to_crash_clusters_over_eighty_meters[
    ["crash_centroid_latitude", "crash_centroid_longitude"]
]

distance = pd.DataFrame(
    [
        [
            geodesic(
                (
                    crash_sites.loc[i, "crash_centroid_latitude"],
                    crash_sites.loc[i, "crash_centroid_longitude"],
                ),
                (
                    crash_sites.loc[j, "crash_centroid_latitude"],
                    crash_sites.loc[j, "crash_centroid_longitude"],
                ),
            ).meters
            for j in crash_sites.index
        ]
        for i in crash_sites.index
    ],
    columns=crash_sites.index,
    index=crash_sites.index,
)

# %%
# from geopy.distance import geodesic


# camera_sites = neartest_cam_to_crash_clusters[['cam_name', 'camera_coordinates_latitude', 'camera_coordinates_longitude']].drop_duplicates().reset_index(drop=True)

# distance = pd.DataFrame([
#     [
#         geodesic((neartest_cam_to_crash_clusters.loc[i, 'crash_centroid_latitude'], neartest_cam_to_crash_clusters.loc[i, 'crash_centroid_longitude']),
#                  (camera_sites.loc[j, 'camera_coordinates_latitude'], camera_sites.loc[j, 'camera_coordinates_longitude'])).meters
#         for j in range(len(camera_sites))
#     ]
#     for i in range(len(neartest_cam_to_crash_clusters))
# ], columns=camera_sites['cam_name'], index=neartest_cam_to_crash_clusters['cluster_id'])

# %%
# setting number of new camera needed

new_cams_needed = st.slider(
    "How many new cameras would you like?",
    1,
    len(neartest_cam_to_crash_clusters_over_eighty_meters),
    10,
)

st.write(
    f"Adding {new_cams_needed} new cameras will cost about ${new_cams_needed*5000}.00"
)

crash_count_dict = neartest_cam_to_crash_clusters_over_eighty_meters[
    "crash_count"
].to_dict()

# %%


# initiate lp
rec_model = pulp.LpProblem("cam_placement_recs", pulp.LpMinimize)

x_matrix = pulp.LpVariable.dicts(
    "assign", (distance.index, distance.columns), 0, 1, cat="Binary"
)
y = pulp.LpVariable.dicts("open", distance.columns, 0, 1, cat="binary")


# equation multiplying crash density by distance from other crash cameras to make sure new camera locations arent stacked
rec_model += pulp.lpSum(
    [
        crash_count_dict[i] * distance.loc[i, j] * x_matrix[i][j]
        for i in distance.index
        for j in distance.columns
    ]
)


# contrains only one cam per cluster
for i in distance.index:
    rec_model += pulp.lpSum([x_matrix[i][j] for j in distance.columns]) == 1

for i in distance.index:
    for j in distance.columns:
        rec_model += x_matrix[i][j] <= y[j]


# contrains camera recommendations to input cameras needed
rec_model += pulp.lpSum([y[j] for j in distance.columns]) == new_cams_needed

rec_model.solve(pulp.PULP_CBC_CMD(msg=False))


# %%
# outputs
cluster_id_recomendations = [j for j in distance.columns if y[j].value() == 1]
assignments = {
    i: [j for j in distance.columns if x_matrix[i][j].value() == 1][0]
    for i in distance.index
}


# %% [markdown]
# Sample Viz

# %%


gdf = ox.geocode_to_gdf({"city": "Washington D.C."})

# %%
cam_locs_coord_only = cam_locatios_geo[["geometry"]]

cam_locs_coord_only["source"] = "cams"

# %%
crash_data_viz = crash_geo_df.merge(cluster_sizes, on="cluster_id", how="left")
crash_locs_coords_only = crash_data_viz[["crash_centroid_coords", "crash_count"]]

crash_locs_coords_only["source"] = "crash"

crash_locs_coords_only = crash_locs_coords_only.rename(
    {"crash_centroid_coords": "geometry"}, axis=1
)

# %%
master_coords = gpd.GeoDataFrame(
    pd.concat([cam_locs_coord_only, crash_locs_coords_only], axis=0), crs="EPSG:4326"
)

master_coords["radius_scaled"] = master_coords["crash_count"].fillna(1)
master_coords["radius_scaled"] = master_coords["radius_scaled"] * 3 * 5

# %%


# %%
new_cams = neartest_cam_to_crash_clusters_over_eighty_meters.loc[
    cluster_id_recomendations
][["crash_centroid_coords_coordinates"]]

# %%
new_cams["source"] = "new_cam"

new_cams = new_cams.rename({"crash_centroid_coords_coordinates": "geometry"}, axis=1)

# %%
master_coords = pd.concat([master_coords, new_cams])

# %%
master_coords["source"].value_counts()

# %%
maps = gdf.explore(name="DC", height=600, width=800)

# %%
# red = camera

maps = master_coords.explore(
    m=maps,
    column="source",
    cmap=["red", "blue", "black"],
    legend=True,
    marker_type="circle",
    radius="radius_scaled",
    tooltips=["crash_count"],
)

# %%
folium.LayerControl().add_to(maps)


st_folium(maps, width=800, height=600)

# %%
# m = geo_df.explore(m=m, column="Outcome", cmap=["red","green","blue"], name="points")


# %% [markdown]
# - get closest camera location to each crash grouping
# - leaflet uses geojson, structure output accordingly
#
#
