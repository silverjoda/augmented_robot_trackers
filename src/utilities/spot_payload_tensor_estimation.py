
def estimate_diagonal_cuboid_tensor(mass, xyz_dims):
    x,y,z = xyz_dims
    xx = mass * (1./12.) * (y ** 2 + z ** 2)
    yy = mass * (1./12.) * (x ** 2 + z ** 2)
    zz = mass * (1./12.) * (x ** 2 + y ** 2)
    return [xx,yy,zz]

def main():
    print(estimate_diagonal_cuboid_tensor(mass=11.9, xyz_dims=(0.7,0.18,0.12)))

if __name__=="__main__":
    main()

# Spot payload config:
# Payload position wrt payload mounting reference frame
# 0,0,0

# Center of mass wrt payload reference frame
# -0.232, 0, 0.092

# Rough mass with everything
# 11.86 kg

# Tensor of inertia estimated as a cuboid, centered at center of mass:
# (Run script)

# Bounding box 1:
# pos: 0.213, 0, 0.1
    # extent: 0.235, 0.09, 0.1

# Bounding box 2:
# pos: 0, 0, 0.22
# extent: 0.18, 0.09, 0.08

# Bounding box 3:
# pos: -0.61, 0, 0.21
# extent: 0.16, 0.09, 0.07