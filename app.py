
import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import linprog
import pydeck as pdk

st.set_page_config(page_title="Smart Transportation Route Optimization", layout="wide")

st.title("🚚 Smart Transportation Route Optimization")
st.markdown("""
This tool helps businesses minimize transportation costs from multiple warehouses to various customer locations using **Operations Research** techniques.

- Upload cost matrix, supply, and demand.
- See optimal delivery plan and total cost.
- Visualize optimized delivery routes on a map.
""")

# Example data with coordinates
def example_data():
    cost_df = pd.DataFrame(
        [[8, 6, 10, 9], [9, 12, 13, 7], [14, 9, 16, 5]],
        columns=["C1", "C2", "C3", "C4"],
        index=["W1", "W2", "W3"]
    )
    supply = [100, 120, 80]
    demand = [60, 70, 90, 80]
    warehouse_coords = {
        "W1": [12.9716, 77.5946],
        "W2": [13.0827, 80.2707],
        "W3": [17.3850, 78.4867]
    }
    customer_coords = {
        "C1": [19.0760, 72.8777],
        "C2": [28.7041, 77.1025],
        "C3": [22.5726, 88.3639],
        "C4": [23.0225, 72.5714]
    }
    return cost_df, supply, demand, warehouse_coords, customer_coords

# Upload or use example
data_option = st.radio("Select Input Method:", ["Use Example Data", "Upload CSV Files"])

if data_option == "Use Example Data":
    cost_df, supply, demand, warehouse_coords, customer_coords = example_data()
    st.subheader("Cost Matrix")
    st.dataframe(cost_df)
else:
    st.subheader("Upload Cost Matrix CSV")
    cost_file = st.file_uploader("Upload Cost Matrix", type=["csv"])
    supply_file = st.file_uploader("Upload Supply CSV", type=["csv"])
    demand_file = st.file_uploader("Upload Demand CSV", type=["csv"])
    wh_coords_file = st.file_uploader("Upload Warehouse Coordinates CSV", type=["csv"])
    cust_coords_file = st.file_uploader("Upload Customer Coordinates CSV", type=["csv"])

    if cost_file and supply_file and demand_file and wh_coords_file and cust_coords_file:
        cost_df = pd.read_csv(cost_file, index_col=0)
        supply = pd.read_csv(supply_file).iloc[:, 1].tolist()
        demand = pd.read_csv(demand_file).iloc[:, 1].tolist()
        wh_coords_df = pd.read_csv(wh_coords_file)
        cust_coords_df = pd.read_csv(cust_coords_file)
        warehouse_coords = dict(zip(wh_coords_df.iloc[:, 0], zip(wh_coords_df.iloc[:, 1], wh_coords_df.iloc[:, 2])))
        customer_coords = dict(zip(cust_coords_df.iloc[:, 0], zip(cust_coords_df.iloc[:, 1], cust_coords_df.iloc[:, 2])))
        st.dataframe(cost_df)
    else:
        st.warning("Please upload all required files.")
        st.stop()

# Prepare linear programming inputs
cost_matrix = cost_df.values
num_sources, num_destinations = cost_matrix.shape

c = cost_matrix.flatten()
A_eq = []
b_eq = []

# Supply constraints
for i in range(num_sources):
    row = [0] * num_sources * num_destinations
    for j in range(num_destinations):
        row[i * num_destinations + j] = 1
    A_eq.append(row)
    b_eq.append(supply[i])

# Demand constraints
for j in range(num_destinations):
    row = [0] * num_sources * num_destinations
    for i in range(num_sources):
        row[i * num_destinations + j] = 1
    A_eq.append(row)
    b_eq.append(demand[j])

bounds = [(0, None) for _ in range(num_sources * num_destinations)]

res = linprog(c=c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

if res.success:
    st.success("✅ Optimization Successful!")
    st.subheader("📊 Optimal Allocation Table")
    result = np.array(res.x).reshape((num_sources, num_destinations))
    result_df = pd.DataFrame(result, index=cost_df.index, columns=cost_df.columns)
    st.dataframe(result_df.style.format("{:.2f}"))
    st.subheader("💰 Total Minimum Transportation Cost")
    st.metric("Cost", f"₹ {res.fun:.2f}")

    # Visualize routes on a map
    st.subheader("🗺️ Optimized Delivery Routes Map")
    route_data = []
    for i, wh in enumerate(cost_df.index):
        for j, cust in enumerate(cost_df.columns):
            if result[i][j] > 0:
                route_data.append({
                    "from_lat": warehouse_coords[wh][0],
                    "from_lon": warehouse_coords[wh][1],
                    "to_lat": customer_coords[cust][0],
                    "to_lon": customer_coords[cust][1],
                    "weight": result[i][j]
                })

    route_df = pd.DataFrame(route_data)

    layer = pdk.Layer(
        "LineLayer",
        route_df,
        get_source_position="[from_lon, from_lat]",
        get_target_position="[to_lon, to_lat]",
        get_width="weight / 10",
        get_color="[0, 100, 255, 160]",
        pickable=True,
    )

    mid_lat = route_df[["from_lat", "to_lat"]].values.mean()
    mid_lon = route_df[["from_lon", "to_lon"]].values.mean()
    view_state = pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=4)

    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state))
else:
    st.error("Optimization failed. Please check your input data.")
