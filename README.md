# Corner detection in 3D LiDAR point clouds
### 0. Loading the data
Notice how the data is at an angle.

![](plots/boss_room_angle.png)

### 1. Detecting edges
By collapsing the data onto two dimensions and Gaussian blurring, a neat contour can be obtained.

![](plots/edge.png)

Using the [Hough transform](https://en.wikipedia.org/wiki/Hough_transform) start and end points denoting lines can be identified. Calculating the vector gradients of these, classification into the two perpendicular wall directions is possible.

![](plots/classified_points.png)

### 2. Identification of subgroups
The subgroups of the two clusters can be further classified using a simple one dimensional density classifier, which I adapted from [DBSCAN](https://en.wikipedia.org/wiki/DBSCAN). 

![](plots/one_dim.png)

Taking averages along their respective dimensions, these subgroups can now be represented with a single parameter in vector lines of the form  $\textbf{x} = \textbf{a} + \lambda \textbf{d}$. 

### 3. Calculating corners
Potential corners can be identified using the linear solutions for the pairwise intersection of any perpendicular lines.

![](/plots/with_corners.png)

A limitation of this algorithm is the presence of supernumerary corners, due to wall indents. Approaches using the [Douglas-Peucker algorithm](https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm) or contour algorithms may filter out the redundant points. However, additional corners may also be of use in room reconstruction.


### 4. Scaling to metres
To convert the point values to metres, we must first rotate the 3d point cloud using [Rodrigues's rotation formula](https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula), the normal, axis and angle.

![](plots/rotated_room.png)

Collapsing to one dimensions in x and y, histogram peaks will indicate the presence of walls.

![](plots/along_x_hist.png)

The x and y differences between the relative positions of these peaks give rise to the dimensions of the room:
`array([3.74314508, 7.08169256])`.

Calculating also the relative differences between the points, the ratio between these two units can be used to scale the corners to metres.

Due to the non-determinism of the probabilistic Hough transform, I have observed ~5cm differences between runs.

```
array([[0.  , 1.1 ],
       [0.29, 1.1 ],
       [6.73, 1.1 ],
       [7.2 , 1.1 ],
       [0.  , 3.99],
       [0.29, 3.99],
       [6.73, 3.99],
       [7.2 , 3.99],
       [0.  , 4.22],
       [0.29, 4.22],
       [6.73, 4.22],
       [7.2 , 4.22],
       [0.  , 4.78],
       [0.29, 4.78],
       [6.73, 4.78],
       [7.2 , 4.78]])
```
