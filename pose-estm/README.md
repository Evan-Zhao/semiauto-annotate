# Realtime Multi-Person Pose Estimation
By [Zhe Cao](https://people.eecs.berkeley.edu/~zhecao/), [Tomas Simon](http://www.cs.cmu.edu/~tsimon/), [Shih-En Wei](https://scholar.google.com/citations?user=sFQD3k4AAAAJ&hl=en), [Yaser Sheikh](http://www.cs.cmu.edu/~yaser/).

## Quick Start
Download pretrained model
```Bash
wget -c http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel -P model/
```

Run it with command
```Bash
python pose_detection.py --input [image_path] --output [output_json_path]
```

The output json is an array of persons' 18 keypoints. Keypoints are also arranged in arrays from 0 to 17.
The limb sequence is 
```python
limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [1, 16], [16, 18], [3, 17], [6, 18]]
```
The corresponding keypoints are 
```c++
const std::string keypointsMapping[] = {
	"Nose", "Neck",
	"R-Sho", "R-Elb", "R-Wr",
	"L-Sho", "L-Elb", "L-Wr",
	"R-Hip", "R-Knee", "R-Ank",
	"L-Hip", "L-Knee", "L-Ank",
	"R-Eye", "L-Eye", "R-Ear", "L-Ear"
};
```

The snippet of draw skeleton in source codes is
```python
    # visualize
    # find connection in the specified sequence, center 29 is in the position 15
    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [1, 16], [16, 18], [3, 17], [6, 18]]
    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    cmap = matplotlib.cm.get_cmap('hsv')

    canvas = cv.imread(test_image)  # B,G,R order

    for i in range(18):
        rgba = np.array(cmap(1 - i / 18. - 1. / 36))
        rgba[0:3] *= 255
        for j in range(len(all_peaks[i])):
            # person_keypoints[j].append(all_peaks[i][j][0:2])
            cv.circle(canvas, all_peaks[i][j][0:2], 4, colors[i], thickness=-1)

    to_plot = cv.addWeighted(oriImg, 0.3, canvas, 0.7, 0)
    plt.imshow(to_plot[:, :, [2, 1, 0]])
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(12, 12)
    plt.show()
    # visualize 2
    stickwidth = 4

    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            cur_canvas = canvas.copy()
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv.fillConvexPoly(cur_canvas, polygon, colors[i])
            canvas = cv.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

    plt.imshow(canvas[:, :, [2, 1, 0]])
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(12, 12)
    plt.show()
```

## Citation
Please cite the paper in your publications if it helps your research:

    
    
    @inproceedings{cao2017realtime,
      author = {Zhe Cao and Tomas Simon and Shih-En Wei and Yaser Sheikh},
      booktitle = {CVPR},
      title = {Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields},
      year = {2017}
      }
	  
    @inproceedings{wei2016cpm,
      author = {Shih-En Wei and Varun Ramakrishna and Takeo Kanade and Yaser Sheikh},
      booktitle = {CVPR},
      title = {Convolutional pose machines},
      year = {2016}
      }
