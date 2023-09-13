"""Generate spherical meshes of various resolutions. Adapted from https://github.com/maxjiang93/ugscnn"""

from gdnn.icosahedron import export_spheres

print("Generating spherical meshes . . . ")
export_spheres(range(8), "mesh_files")

print("Done!")
