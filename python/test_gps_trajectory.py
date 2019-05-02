import sys
sys.path.append("../")
from utils.preprocessing import read_filenames
import pandas
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


# Read in gps file and parse it to numpy array
def parse_gpsfile(file_path):
    """
    :param file_path: The path to the specified gps file (in csv format)
    :return: the parsed gps data (in dictionary format)
    """
    # Read file from csv
    data = pandas.read_csv(file_path)

    # Convert to dictionary
    data_array = data.values
    gps_data = {
        "timestamps": data_array[:, 1],
        "px": data_array[:, 2],
        "py": data_array[:, 3],
        "pz": data_array[:, 4],
        "qx": data_array[:, 5],
        "qy": data_array[:, 6],
        "qz": data_array[:, 7],
        "qw": data_array[:, 8]
    }

    return gps_data


# Plot list of trajectories
def plot_trajectory(gps_files):
    """
    :param gps_files: a single or a list of parsed gps data (sess parse_gpsfile() for more details)
    :return: None => only plot the trajectories
    """
    assert len(gps_files) > 0

    # Create figure
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_zlim3d(0, 200)

    if isinstance(gps_files, list):
        # Plot all the input sequences in gps file
        for idx, data in enumerate(gps_files):
            # Plot single trajectory
            if data['px'].shape[0] == 0:
                continue
            ax.plot3D(data["px"], data["py"], data["pz"], label="seq%02d"%(idx),
                      scalex=True, scaley=True, marker='o')

    elif isinstance(gps_files, dict):
        # Plot single trajctory
        plot_range = range(810, 850)
        ax.plot3D(gps_files["px"][plot_range], gps_files["py"][plot_range], gps_files["pz"][plot_range], label="seq%00",
                  scalex=True, scaley=True, marker='o')

    else:
        raise ValueError("Unsupported type!!")

    ax.legend()
    plt.show()


if __name__ == "__main__":
    name_dataset = read_filenames()

    # with open("cached_filenames.pkl", "wb") as f:
    #     pickle.dump(name_dataset, f)

    with open("cached_filenames.pkl", "rb") as f:
        pickle.load(f)

    # Read gps data
    gps_path_lst = []
    for seq_name in name_dataset.keys():
        gps_path_lst.append(name_dataset[seq_name]["gps"])

    seq_name_lst = list(name_dataset.keys())
    gps_data_lst = [parse_gpsfile(_) for _ in gps_path_lst]

    # Plot the trajectory
    # ToDo: /cluster/work/riner/users/PLR-2019/lin/dataset_ASL/20180907_092223_merged_cds1_vs/gps_utm.csv
    # ToDo: 810~850 have abnormal jumps
    plot_trajectory(gps_data_lst[0:10])