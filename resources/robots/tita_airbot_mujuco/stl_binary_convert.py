from stl import mesh
import os

ascii_path = "/home/jks_n/桌面/DDT_VE/pointfoot-legged-gym/resources/robots/tita_airbot_mujuco/urdf/meshes/arm/airbot_play_v3_1/visual/link2.STL"
binary_path = "/home/jks_n/桌面/DDT_VE/pointfoot-legged-gym/resources/robots/tita_airbot_mujuco/urdf/meshes/arm/airbot_play_v3_1/visual/link2_bin.STL"

your_mesh = mesh.Mesh.from_file(ascii_path)
your_mesh.save(binary_path, mode=mesh.stl.Mode.BINARY)
