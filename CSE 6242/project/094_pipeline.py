
import math
import json
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import geopandas as gpd
import pulp
import dash
from dash import dcc, html, Input, Output, State, dash_table
from sklearn.metrics.pairwise import haversine_distances
import plotly.graph_objects as go


cam_locations = pd.read_csv(
    r"data/Automated_Traffic_Enforcement_Table.csv", low_memory=False
)
crash_data = pd.read_csv(r"data/Crashes_in_DC.csv", low_memory=False)

precomp = {}
ready_for_compute = False


def precompute():
    global precomp, ready_for_compute, crash_data, cam_locations

    crash_data = crash_data[crash_data["SPEEDING_INVOLVED"] >= 1]

    crash_data["REPORTDATE"] = pd.to_datetime(crash_data["REPORTDATE"])
    crash_data_filtered = crash_data[
        crash_data["REPORTDATE"].between("2010-01-01", "2025-10-17", inclusive="both")
    ]
    crash_data_filtered = crash_data_filtered.rename(
        {"LATITUDE": "latitude", "LONGITUDE": "longitude"}, axis=1
    )

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

    crash_data_filtered = crash_data_filtered.rename(
        {"LATITUDE": "latitude", "LONGITUDE": "longitude"}, axis=1
    )

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

    # used 80m for ~standard city block size
    crash_clusters = cluster_crashes(crash_data_filtered, 10, eps_meters=80)

    # crash density per cluster
    cluster_sizes = (
        crash_clusters.groupby("cluster_id", as_index=False)
        .size()
        .rename({"size": "crash_count"}, axis=1)
    )

    crash_geo_df = gpd.GeoDataFrame(
        crash_clusters,
        geometry=gpd.points_from_xy(
            crash_clusters["longitude"], crash_clusters["latitude"]
        ),
    )

    # getting central point for each cluster
    cluster_centroids = (
        crash_geo_df[crash_geo_df["cluster_id"] != -1]
        .dissolve(by="cluster_id")
        .centroid
    )

    crash_cluster_centroids = cluster_centroids.reset_index(drop=False)
    crash_cluster_centroids.columns = ["cluster_id", "crash_centroid_coords"]

    crash_cluster_centroids["cluster_id"].duplicated().sum()

    crash_geo_df = gpd.GeoDataFrame(
        crash_cluster_centroids, geometry="crash_centroid_coords", crs="EPSG:4326"
    )

    # getting geodf for cam locations
    cam_locatios_geo = gpd.GeoDataFrame(
        cam_data_w_coordinates,
        geometry=gpd.points_from_xy(
            cam_data_w_coordinates["CAMERA_LONGITUDE"],
            cam_data_w_coordinates["CAMERA_LATITUDE"],
        ),
        crs="EPSG:4326",
    )[["ENFORCEMENT_SPACE_CODE", "geometry"]]

    cam_locations_proj = cam_locatios_geo.to_crs(epsg=26918)
    crash_locations_proj = crash_geo_df.to_crs(epsg=26918)

    # getting nearest cam to crash clusters
    nearest_cameras = gpd.sjoin_nearest(
        crash_locations_proj, cam_locations_proj, distance_col="distance_m"
    )
    nearest_cameras = nearest_cameras.sort_values(
        by="distance_m", ascending=True
    ).drop_duplicates(subset="cluster_id", keep="first")

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
    ).drop(columns=["index_right"], errors="ignore")

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

    # filtering for proximity to current camera locations
    neartest_cam_to_crash_clusters_over_eighty_meters = neartest_cam_to_crash_clusters[
        neartest_cam_to_crash_clusters["nearst_camera_distance"] > 80
    ]

    neartest_cam_to_crash_clusters_over_eighty_meters = (
        neartest_cam_to_crash_clusters_over_eighty_meters.set_index("cluster_id")
    )

    # calculating distance between crash clusters, running slow room to improve
    crash_sites = neartest_cam_to_crash_clusters_over_eighty_meters[
        ["crash_centroid_latitude", "crash_centroid_longitude"]
    ]
    if crash_sites.empty:
        precomp.update(
            {
                "distance": pd.DataFrame(),
                "crash_sites": pd.DataFrame(),
                "crash_count_dict": {},
                "addr_map": pd.DataFrame(),
                "cluster_pts_source": pd.DataFrame(
                    columns=["cluster_id", "latitude", "longitude"]
                ),
                "cam_locs_coord_only": gpd.GeoDataFrame(columns=["geometry"]),
            }
        )
        ready_for_compute = True
        return

    coords_rad = np.radians(
        crash_sites[["crash_centroid_latitude", "crash_centroid_longitude"]].to_numpy()
    )
    distance = haversine_distances(coords_rad) * 6371000
    distance = pd.DataFrame(
        distance, index=crash_sites.index, columns=crash_sites.index
    )

    crash_count_dict = neartest_cam_to_crash_clusters_over_eighty_meters[
        "crash_count"
    ].to_dict()


    cam_locs_coord_only = gpd.GeoDataFrame(
        cam_data_w_coordinates,
        geometry=gpd.points_from_xy(
            cam_data_w_coordinates["CAMERA_LONGITUDE"],
            cam_data_w_coordinates["CAMERA_LATITUDE"],
        ),
        crs="EPSG:4326",
    )[["geometry"]]
    cam_locs_coord_only["source"] = "cams"
    cluster_pts_source = crash_clusters[["cluster_id", "latitude", "longitude"]].copy()

    precomp.update(
        {
            "neartest_cam_to_crash_clusters_over_eighty_meters": neartest_cam_to_crash_clusters_over_eighty_meters,
            "crash_sites": crash_sites,
            "distance": distance,
            "crash_count_dict": crash_count_dict,
            "cluster_pts_source": cluster_pts_source,
            "cam_locs_coord_only": cam_locs_coord_only,
            "crash_points_all": crash_clusters[
                ["cluster_id", "latitude", "longitude"]
            ].copy(),
        }
    )
    ready_for_compute = True


precompute()

# Load the GeoJSON data
with open("data/Automated_Traffic_Enforcement_Table.geojson", "r") as f:
    geojson_data = json.load(f)

# Extract features and filter for active/live cameras with valid coordinates
camera_data = []
for feature in geojson_data["features"]:
    props = feature["properties"]
    if (
        props["ACTIVE_STATUS"] == "Active"
        and props["CAMERA_STATUS"] == "Live"
        and props["CAMERA_LATITUDE"] is not None
        and props["CAMERA_LONGITUDE"] is not None
    ):
        camera_data.append(
            {
                "Location": props["LOCATION_DESCRIPTION"],
                "Latitude": props["CAMERA_LATITUDE"],
                "Longitude": props["CAMERA_LONGITUDE"],
                "Enforcement Type": props["ENFORCEMENT_TYPE"],
                "Speed Limit": props["SPEED_LIMIT"],
                "Ward": props["WARD"],
                "Start Date": props["START_DATE"],
                "Device Mobility": props["DEVICE_MOBILITY"],
            }
        )
# Stoplight cost
cost_per_light = 80000

# Create DataFrame from extracted data
df = pd.DataFrame(camera_data)


def compute_recommendations(cams_needed):
    distance = precomp.get("distance", pd.DataFrame())
    crash_sites = precomp.get("crash_sites", pd.DataFrame())
    crash_count_dict = precomp.get("crash_count_dict", {})
    addr_map = precomp.get("addr_map", pd.DataFrame())
    neartest_cam_to_crash_clusters_over_eighty_meters = precomp.get(
        "neartest_cam_to_crash_clusters_over_eighty_meters", pd.DataFrame()
    )
    cam_locs_coord_only = precomp.get("cam_locs_coord_only", gpd.GeoDataFrame())
    cluster_pts_source = precomp.get("cluster_pts_source", pd.DataFrame())

    if distance.empty or crash_sites.empty:
        df_recs = pd.DataFrame(columns=["latitude", "longitude"])
        df_cams = pd.DataFrame(columns=["latitude", "longitude"])
        df_tab = pd.DataFrame(
            columns=[
                "cluster_id",
                "latitude",
                "longitude",
                "crash_address",
                "crash_count",
            ]
        )
        cluster_pts = pd.DataFrame(columns=["cluster_id", "latitude", "longitude"])
        return df_recs, df_cams, df_tab, cluster_pts

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
    rec_model += pulp.lpSum([y[j] for j in distance.columns]) == cams_needed

    rec_model.solve(pulp.PULP_CBC_CMD(msg=False))

    # outputs
    cluster_id_recomendations = [j for j in distance.columns if y[j].value() == 1]

    max_lookups = 15
    sites_for_recs = crash_sites.loc[cluster_id_recomendations].head(max_lookups).copy()


    new_cams = neartest_cam_to_crash_clusters_over_eighty_meters.loc[
        cluster_id_recomendations
    ][["crash_centroid_coords_coordinates"]]

    new_cams["crash_count"] = new_cams.index.map(crash_count_dict)


    new_cams["source"] = "new_cam"

    new_cams = new_cams.rename(
        {"crash_centroid_coords_coordinates": "geometry"}, axis=1
    )

    df_tab = new_cams.copy()
    df_tab["latitude"] = df_tab["geometry"].y
    df_tab["longitude"] = df_tab["geometry"].x
    df_tab["crash_count"] = df_tab["crash_count"].fillna(0).astype(int)
    df_tab = (
        df_tab[["latitude", "longitude"]]
        .join(pd.Series(df_tab.index, name="cluster_id"))[
            ["cluster_id", "latitude", "longitude"]
        ]
        .join(df_tab[["crash_count"]])
    )
    df_tab["cluster_id"] = pd.to_numeric(df_tab["cluster_id"], errors="coerce").astype(
        "Int64"
    )
    df_tab["crash_count"] = (
        pd.to_numeric(df_tab["crash_count"], errors="coerce").fillna(0).astype("Int64")
    )

    try:
        if "new_cams" in locals():
            df_new = new_cams.copy()
            if "geometry" in df_new.columns:
                df_new["latitude"] = df_new["geometry"].y
                df_new["longitude"] = df_new["geometry"].x
                df_recs = (
                    df_new[["latitude", "longitude"]].dropna().reset_index(drop=True)
                )
  
        else:
            df_recs = pd.DataFrame(columns=["latitude", "longitude"])
    except Exception as e:
        df_recs = df_recs = pd.DataFrame(columns=["latitude", "longitude"])

    try:
        if "cam_locs_coord_only" in locals():
            cams = cam_locs_coord_only.copy()
            if "geometry" in cams.columns:
                cams["latitude"] = cams["geometry"].y
                cams["longitude"] = cams["geometry"].x
                df_cams = (
                    cams[["latitude", "longitude"]].dropna().reset_index(drop=True)
                )
            else:
                df_cams = pd.DataFrame(columns=["latitude", "longitude"])
        else:
            df_cams = pd.DataFrame(columns=["latitude", "longitude"])
    except Exception as e:
        df_cams = pd.DataFrame(columns=["latitude", "longitude"])

    crash_points_all = precomp.get("crash_points_all", pd.DataFrame())
    if crash_points_all.empty:
        cluster_pts = pd.DataFrame(columns=["cluster_id", "latitude", "longitude"])
    else:
        cluster_pts = crash_points_all[
            crash_points_all["cluster_id"].isin(cluster_id_recomendations)
        ].copy()

    return df_recs, df_cams, df_tab, cluster_pts


# Initialize Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Define app layout
app.layout = html.Div(
    [
        html.H3(
            "Optimizing Traffic Cameras in D.C. to Reduce Crashes",
            style={"textAlign": "center"},
        ),
        html.H5("Group 094", style={"textAlign": "center"}),
        dcc.Checklist(
            id="camera-toggle",
            options=[{"label": "Show Cameras", "value": "show"}],
            value=["show"],
            style={"textAlign": "center", "margin": "10px"},
        ),
        dcc.Checklist(
            id="cluster-toggle",
            options=[{"label": "Show Full Cluster", "value": "show_full_cluster"}],
            value=[],
            style={"textAlign": "center", "margin": "0 10px 10px"},
        ),
        html.Div(
            [
                html.Label("Budget ($)"),
                dcc.Input(
                    id="budget",
                    type="text",
                    value=0,
                    min=0,
                    inputMode="numeric",
                    debounce=True,
                    style={"width": "200px"},
                ),
                html.Button(
                    "Update", id="run_button", n_clicks=0, style={"marginLeft": "12px"}
                ),
            ],
            style={
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "center",
                "gap": "8px",
                "marginBottom": "8px",
            },
        ),
        html.Div(id="output_text", style={"textAlign": "center", "marginTop": "8px"}),
        dcc.Graph(id="map"),
        html.Div(
            id="cluster_table", style={"maxWidth": "1100px", "margin": "12px auto"}
        ),
        html.Div(
            [
                html.Button(
                    "Download CSV",
                    id="button_download",
                    n_clicks=0,
                    style={"margin": "8px 0"},
                ),
                dcc.Download(id="download_table"),
                dcc.Store(id="cluster_table_storage"),
            ],
            style={"maxWidth": "1100px", "margin": "0 auto"},
        ),
    ],
    style={
        "background": "linear-gradient(135deg, #1a1a1a, #3a3a3a)",
        "color": "white",
        "minHeight": "100vh",
        "padding": "20px",
    },
)


@app.callback(
    [
        Output("map", "figure"),
        Output("output_text", "children"),
        Output("cluster_table", "children"),
        Output("cluster_table_storage", "data"),
    ],
    Input("run_button", "n_clicks"),
    Input("camera-toggle", "value"),
    Input("cluster-toggle", "value"),
    State("budget", "value"),
)
def update_map(n_clicks, camera_toggle, cluster_toggle, budget):
    try:
        raw = (budget or "").replace(",", "").strip()
        budget_val = float(raw) if raw not in ("", None) else 0.0
    except Exception:
        budget_val = 0.0
    budget = budget_val
    cams_needed = int(max(0, math.floor(budget / float(cost_per_light))))

    if 0 < budget < cost_per_light:
        output_text = html.Div(
            f"Insufficient budget, minimum budget is ${cost_per_light:,.0f}",
            style={"color": "red", "fontWeight": "bold"},
        )
        fig = go.Figure()
        fig.update_layout(
            mapbox_style="open-street-map",
            mapbox_center={"lat": 38.8951, "lon": -77.0364},
            mapbox_zoom=11,
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            height=600,
        )
        tabulars = html.Div("No Information to Display")
        return fig, output_text, tabulars, []

    show_camera = bool(camera_toggle and "show" in camera_toggle)

    if cams_needed == 0:
        df_recs = pd.DataFrame(columns=["latitude", "longitude"])
        df_cams = pd.DataFrame(columns=["latitude", "longitude"])
        df_tab = pd.DataFrame(
            columns=["cluster_id", "latitude", "longitude", "crash_count"]
        )
        cluster_pts = pd.DataFrame(columns=["cluster_id", "latitude", "longitude"])
    else:
        df_recs, df_cams, df_tab, cluster_pts = compute_recommendations(cams_needed)

    if df_tab is None or df_tab.empty:
        tabulars = html.Div("No Information to Display")
        table_store_data = []
    else:
        tabular_view = df_tab[
            ["cluster_id", "latitude", "longitude", "crash_count"]
        ].copy()

        tabular_view["cluster_id"] = pd.to_numeric(
            tabular_view["cluster_id"], errors="coerce"
        ).astype("Int64")
        tabular_view["latitude"] = pd.to_numeric(
            tabular_view["latitude"], errors="coerce"
        ).round(6)
        tabular_view["longitude"] = pd.to_numeric(
            tabular_view["longitude"], errors="coerce"
        ).round(6)
        tabular_view["crash_count"] = pd.to_numeric(
            tabular_view["crash_count"], errors="coerce"
        ).astype("Int64")

        cols = [
            {"name": "latitude", "id": "latitude", "type": "numeric"},
            {"name": "longitude", "id": "longitude", "type": "numeric"},
            {"name": "crash count", "id": "crash_count", "type": "numeric"},
        ]
        tabular_view = tabular_view.sort_values("crash_count", ascending=False)
        tabulars = dash_table.DataTable(
            id="cluster_table_dt",
            columns=cols,
            data=tabular_view.to_dict("records"),
            page_size=15,
            sort_action="native",
            filter_action="native",
            row_selectable="single",
            selected_rows=[],
            style_table={
                "overflowX": "auto",
                "border": "1px solid #444",
                "borderRadius": "10px",
                "backgroundColor": "#2c2c2c",
                "padding": "5px",
                "marginTop": "10px",
            },
            style_cell={
                "fontFamily": "Arial, sans-serif",
                "fontSize": "14px",
                "padding": "8px",
                "backgroundColor": "#2c2c2c",
                "color": "#f1f1f1",
                "border": "1px solid #444",
                "textAlign": "center",
                "whiteSpace": "normal",
                "height": "auto",
            },
            style_header={
                "backgroundColor": "#444",
                "color": "white",
                "fontWeight": "bold",
                "border": "1px solid #555",
                "textAlign": "center",
            },
            style_data_conditional=[
                {"if": {"row_index": "odd"}, "backgroundColor": "#333"},
                {"if": {"column_id": "cluster_id"}, "textAlign": "right"},
                {"if": {"column_id": "crash_count"}, "textAlign": "right"},
                {"if": {"column_id": "latitude"}, "textAlign": "right"},
                {"if": {"column_id": "longitude"}, "textAlign": "right"},
                {
                    "if": {"state": "selected"},
                    "backgroundColor": "#666",
                    "color": "white",
                    "border": "1px solid #aaa",
                },
            ],
            style_as_list_view=False,
        )
        table_store_data = tabular_view.to_dict("records")

    fig = go.Figure()

    if show_camera and not df.empty:
        fig.add_trace(
            go.Scattermap(
                lat=df["Latitude"],
                lon=df["Longitude"],
                mode="markers",
                marker=dict(size=9),
                name="Cameras (Active/Live)",
            )
        )
    show_clusters = cluster_toggle and "show_full_cluster" in cluster_toggle
    if show_clusters and cluster_pts is not None and not cluster_pts.empty:
        fig.add_trace(
            go.Scattermap(
                lat=cluster_pts["latitude"],
                lon=cluster_pts["longitude"],
                mode="markers",
                marker=dict(size=9, color="red", opacity=0.4),
                name="Cluster Points",
            )
        )
    elif df_recs is not None and not df_recs.empty:
        fig.add_trace(
            go.Scattermap(
                lat=df_recs["latitude"],
                lon=df_recs["longitude"],
                mode="markers",
                marker=dict(size=12),
                name=f"Recommended",
            )
        )
    if len(fig.data) == 0:
        fig.add_trace(
            go.Scattermap(
                lat=[38.8951],
                lon=[-77.0364],
                mode="markers",
                marker={"size": 1, "opacity": 0},
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.update_layout(
        map=dict(
            style="open-street-map", center={"lat": 38.8951, "lon": -77.0364}, zoom=11
        ),
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        height=600,
    )
    output_text = (
        f"Budget: ${budget:,.0f} will allow for an additional {cams_needed} camera(s)"
    )

    return fig, output_text, tabulars, table_store_data


@app.callback(
    Output("map", "figure", allow_duplicate=True),
    Input("cluster_table_dt", "selected_rows"),
    State("cluster_table_dt", "derived_virtual_data"),
    State("map", "figure"),
    prevent_initial_call=True,
)
def highlight_selected_points(selected_rows, derived_data, current_fig):
    import plotly.graph_objects as go

    if not derived_data or not selected_rows:
        return current_fig
    row_ind = selected_rows[0]
    if row_ind < 0 or row_ind >= len(derived_data):
        return current_fig
    row = derived_data[row_ind]
    lat, lon = row.get("latitude"), row.get("longitude")
    if lat is None or lon is None:
        return current_fig

    fig = go.Figure(current_fig)

    fig.data = tuple(
        t for t in fig.data if getattr(t, "name", None) != "Selected Point"
    )

    fig.add_trace(
        go.Scattermap(
            lat=[lat],
            lon=[lon],
            mode="markers",
            marker=dict(size=26, color="#FFD700", symbol=["star"], opacity=1),
            name="Selected Point",
            below=None,
        )
    )
    return fig


@app.callback(
    Output("download_table", "data"),
    Input("button_download", "n_clicks"),
    State("cluster_table_storage", "data"),
    prevent_initial_call=True,
)
def download_table(n_clicks, rows):
    if not rows:
        return dash.no_update
    df = pd.DataFrame(rows)
    cols = ["latitude", "longitude", "crash_count"]
    cols = [c for c in cols if c in df.columns]
    df = df[cols]
    return dcc.send_data_frame(df.to_csv, "cluster_recommendations.csv", index=False)


if __name__ == "__main__":
    app.run(port=8070, debug=True)
