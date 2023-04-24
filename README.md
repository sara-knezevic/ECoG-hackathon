<div id="user-content-toc">
  <ul>
    <summary>
      <h1>VIRTUAL BR41N.IO HACKATHON - g.tec Spring School 2023</h1>
      <br>1st place winner for data analysis projects!</br>
    </summary>
  </ul>
</div>

### Problem statement

Hand pose recognition has various applications in fields such as robotics, virtual reality, and medical prosthetics. 
ECoG signals are a type of neural signal that is recorded by placing electrodes directly on the surface of the brain. 
ECoG signals have high temporal and spatial resolution and are commonly used for brain-computer interface (BCI) applications.

The challenge in this project for this hackathon is to develop a system that can accurately recognize hand poses from ECoG signals.
The system should be able to classify hand poses - fist movement [1], peace movement [2], open hand [3]. 
After classification, integers denoting hand poses are sent to a prosthetic arm to display the hand position [Code TBA].

### Dataset

The dataset for the hackathon was provided by <a href="https://www.gtec.at/">g.tec medical engineering GmbH</a> for the hackathon and it includes a
matlab matrix of ECoG data of one patient. It includes ECoG signals recorded from multiple channels, along with the corresponding hand pose labels.

### Approach & resources

Submitted work can be found in the file ECoG_hands.ipynb where the following is done: 
* Conversion of the data into an MNE object
* Preprocessing and filtering
* Pipelines including feature selection and training with several different models

Feature selection algorithms (not conclusive): common spatial patterns (CSP), riemann spatial covariances.<br>
Models trained (not conclusive): LDA, random forest, SVM.

Libraries used for ECoG data:
* MNE
* SciPy
* scikit-learn
* pyriemann
