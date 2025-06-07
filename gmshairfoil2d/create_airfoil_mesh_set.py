import glob
import os

from mesh_function import create_parent_mesh, mesh_airfoil


def create_mesh_set():
    input_directory = "/mnt/c/Users/patbo/Documents/GitHub/airfoil_tools/tmp/single_foil_test"

    airfoil_paths = [airfoil_path for airfoil_path in glob.glob(os.path.join(input_directory,"*airfoil_output*"))]

    for airfoil_iteration, airfoil_path in enumerate(airfoil_paths):
        print(airfoil_path)
        airfoil_number =int(os.path.basename(airfoil_path).split("_")[-1].split(".txt")[0]) 
        airfoil_output_name = f"s1223_design_SA_{airfoil_number}"

        mesh_airfoil(
            output_path=input_directory, 
            output_name=airfoil_output_name,
            local_airfoil=airfoil_path,
            airfoil_mesh_size = 0.003, 
            ratio=1.2,
            no_bl=False,
            first_layer=1.5e-5,
            nb_layers=26,
            farfield=20,
            ext_mesh_size=1.5,
            wake_size_start=0.002,
            wake_size_end=0.6,
            )
        if airfoil_iteration >= 10:
            break

def _create_parent_mesh():
    input_directory = "/mnt/c/Users/patbo/Documents/GitHub/airfoil_tools/tmp/single_foil_test"

    airfoil_output_name = "parent_mesh"


    create_parent_mesh(
        output_path=input_directory, 
        output_name=airfoil_output_name,
        farfield=20,
        ext_mesh_size=1.5,
        wake_size_start=0.02,
        wake_size_end=1,
        )

if __name__ == "__main__":
    _create_parent_mesh()