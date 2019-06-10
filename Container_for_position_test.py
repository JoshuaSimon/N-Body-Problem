# Container_for_position_test.py

# Input Data
# In our case r1_start would be of type vector.
# For demonstration tupel and list is used here.

#r1_start = (0,1)
#r2_start = (0,0)

r1_start = [0,1]
r2_start = [0,0]



# R contains all position data. Initialized with start positions.
R = [[r1_start], [r2_start]]

print("R at start: ", R)

# Number of bodys
n = 2

# Append R with new positions calculated by wheeler algorithm
for i in range(n):
    
    # Appending just some random position values
    #r_new = (i,i)
    r_new = [i,i]

    R[i].append(r_new)

    print("R at", i, "is", R)


print()
print(R[0])                 # A list with all position vectors of one body in R
print(R[0][0])              # One position vector of one body in R
print(R[0][0][0])           # Single value component of a position vector in R