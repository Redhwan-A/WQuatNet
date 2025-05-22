# WQuatNet
WQuatNet: Wide range quaternion-based head pose estimation



# Results visualization


<table>
<tr>
<td><img src="images/cmu1.jpg" height="160"></td>
<td><img src="images/cmu4.jpg" height="160"></td> 
<td><img src="images/cmu13.jpg" height="160"></td> 
<td><img src="images/cmu14.jpg" height="160"></td> 
<td><img src="images/cmu15.jpg" height="160"></td>
<td><img src="images/cmu18.jpg" height="160"></td> 
<td><img src="images/cmu20.jpg" height="160"></td> 
</tr>
</table>

* **Fig.** Snapshots of 7 different views from one sequence in the CMU Panoptic val-set dataset.





## **Our results**
* **Trained and tested on CMU Panoptic datasets.**

| Method          | Retrain? | Yaw      | Pitch    | Roll     | MAE      |
| --------------- | -------- | -------- | -------- | -------- | -------- |
| DirectMHP       | Yes      | 5.86     | 8.25     | 7.25     | 7.12     |
| DirectMHP       | No       | 5.75     | 8.01     | 6.96     | 6.91     |
| 6DRepNet        | Yes      | 5.20     | 7.22     | 6.00     | 6.14     |
| 6DoF-HPE (ours) | Yes      | **5.13** | **6.99** | **5.77** | **5.96** |
| WQuatNet (ours) | Yes      | **4.67** | **6.77** | **5.42** | **5.62** |


| Method          | Retrain? | Yaw      | Pitch    | Roll     | MAE      |
| --------------- | -------- | -------- | -------- | -------- | -------- |
| Viet et al.     | No       | 9.55     | 11.29    | 8.32     | 9.72     |
| DirectMHP       | Yes      | 7.38     | 8.56     | 7.47     | 7.80     |
| DirectMHP       | No       | 7.32     | 8.54     | 7.35     | 7.74     |
| WHENet          | No       | 8.51     | 7.67     | 6.78     | 7.65     |
| Cobo et al.     | No       | –        | –        | –        | 7.45     |
| 6DRepNet        | Yes      | 5.89     | 7.76     | 6.39     | 6.68     |
| 6DoF-HPE (ours) | Yes      | **5.83** | **7.63** | 6.35     | **6.60** |
| WQuatNet (ours) | Yes      | 6.04     | 7.72     | **6.34** | 6.70     |


| Method         | Retrain? | Yaw   | Pitch | Roll  | MAE   |
|----------------|----------|-------|-------|-------|-------|
| WHENet         | No       | 37.96 | 22.7  | 16.54 | 25.73 |
| HopeNet        | No       | 20.40 | 17.47 | 13.40 | 17.09 |
| FSA-Net        | No       | 17.52 | 16.31 | 13.02 | 15.62 |
| Img2Pose       | No       | 12.99 | 16.69 | 15.64 | 15.11 |
| DirectMHP      | Yes      | 5.86  | 8.25  | 7.25  | 7.12  |
| DirectMHP      | No       | 5.75  | 8.01  | 6.96  | 6.91  |
| 6DRepNet       | Yes      | 5.20  | 7.22  | 6.00  | 6.14  |
| 6DoF-HPE (ours)  | Yes      | **5.13**  | **6.99**  | **5.77**  | **5.96** |
| WHENet         | No       | 29.87 | 19.88 | 14.66 | 21.47 |
| DirectMHP      | Yes      | 7.38  | 8.56  | 7.47  | 7.80  |
| DirectMHP      | No       | 7.32  | 8.54  | 7.35  | 7.74  |
| 6DRepNet       | Yes      | 5.89  | 7.76  | 6.39  | 6.68  |
| 6DoF-HPE (ours)  | Yes      | **5.83**  | **7.63**  | **6.35**  | **6.60**  |


* **Trained on 300W-LP, and then test on AFLW2000 and BIWI.**

| Method         | Retrain? | AFLW2000 Yaw | AFLW2000 Pitch | AFLW2000 Roll | AFLW2000 MAE | BIWI Yaw | BIWI Pitch | BIWI Roll | BIWI MAE |
|----------------|----------|--------------|----------------|---------------|-------------|----------|------------|-----------|----------|
| HopeNet        | No       | 6.47         | 6.56           | 5.44          | 6.16        | 4.81     | 6.60       | 3.27      | 4.89     |
| FSA-Net        | No       | 4.50         | 6.08           | 4.64          | 5.07        | 4.27     | 4.96       | 2.76      | 4.00     |
| WHENet         | No       | 4.44         | 5.75           | 4.31          | 4.83        | 3.60     | 4.10       | 2.73      | 3.48     |
| LSR                        | No       | 4.26         | 5.27           | 3.89         | 4.47         | 4.29     | **3.09**       | 3.18      | 3.52     |
| DirectMHP      | No       | **2.99**         | 5.35           | 3.77          | 4.04        | 3.57     | 5.47       | 4.02      | 4.35     |
| 6DRepNet       | No       | 3.63         | 4.91           | 3.37          | 3.97        | **3.24**     | 4.48       | **2.68**     | **3.47**     |
| Img2Pose       | No       | 3.43         | 5.03           | **3.28**          | 3.91        | 4.57     | 3.55       | 3.24      | 3.79     |
| DirectMHP      | Yes      | 3.31         | 5.36           | 3.75          | 4.14        | 3.54     | 5.45       | 4.01      | 4.33     |
| 6DRepNet       | Yes      | 3.50         | 4.81           | 3.47          | 3.93        | 3.79     | 4.53       | 2.89      | 3.74     |
| 6DoF-HPE (ours)  | Yes      | 3.56         | **4.74**           | 3.35          | **3.88**       | 3.91     | 4.43       | 2.69      | 3.68     |


# Datasets

* **CMU Panoptic**  from [here](http://domedb.perception.cs.cmu.edu/) for the full range angles.
  
* **300W-LP**, and **AFLW2000** from [here](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm) for the narrow range angles.

* **BIWI**  from [here](https://icu.ee.ethz.ch/research/datsets.html) for the narrow range angles.

  



## **Training for rotational (yaw, pitch, and roll)  components**:

If you **only** need to change the pre-trained RepVGG model '**RepVGG-B1g4-train.pth**' please see [here](https://drive.google.com/drive/folders/1Avome4KvNp0Lqh2QwhXO6L5URQjzCjUq) and save it in the root directory.


```
python3 train.py
```

After training is done. Next step.

##  **Deploy models for rotational (yaw, pitch, and roll)  components**:

For reparameterization, the trained models into inference models use the convert script.

```
python3 convert.py input-model.tar output-model.pth
```

After converting the training model into an inference model. 
Then, you can test your model.


## **Testing for rotational (yaw, pitch, and roll)  components**:

```
python3 test.py
```



# Citing

```
@article{algabri2025wquatnet,
  title={WQuatNet: Wide range quaternion-based head pose estimation},
  author={Algabri, Redhwan and Shin, Hyunsoo and Abdu, Ahmed and Bae, Ji-Hun and Lee, Sungon},
  journal={Journal of King Saud University Computer and Information Sciences},
  volume={37},
  number={3},
  pages={24},
  year={2025},
  publisher={Springer}
}
```
