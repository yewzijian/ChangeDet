# City-scale Scene Change Detection using Point Clouds
This "src/" folder in this repository contains some useful codes for the following paper:
```
@inproceedings{yew2020-CityScaleCD, 
    title={City-scale Scene Change Detection using Point Clouds}, 
    author={Yew, Zi Jian and Lee, Gim Hee}, 
    booktitle={International Conference on Robotics and Automation (ICRA)},
    year={2021} 
}
```

## Visualize annotations

Download the annotated images from the [project website](https://yewzijian.github.io/ChangeDet/), place them in the "eval_data/" folder. Then run the following code to visualize the images:

```
python src/visualize.py
```

Disappearance and appearance of objects are marked in blue and red respectively.

## Projection and Evaluation code

1. Ensure that our annotated images are placed in the "eval_data/" folder.

2. Download our detected changes, and place them in the "results3d/" folder.

3. Project the detected changes onto the 2D test images, by running:

   ```
   python src/project_changes.py
   ```

   The projected changes will be placed in the "results/" folder.
4. Lastly, run the evaluation code to compute the evaluation metrics.

   ```
   python evaluate.py
   ```

