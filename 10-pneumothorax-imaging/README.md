### SIIM-ACR Pneumothorax Segmentation

* Identify Pneumothorax disease in chest x-rays 
  + https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/overview

### Introduction
_____________________________________________________________________________________________

#### What is pneumothorax?

Imagine suddenly gasping for air, helplessly breathless for no apparent reason. Could it be a collapsed lung?

A pneumothorax occurs when air leaks into the space between your lung and chest wall called the pleura. Imagine a small layer surrounding the lung and that is the pleura cavity. This air pushes on the outside of your lung and makes it collapse. Pneumothorax can be a complete lung collapse or a collapse of only a portion of the lung. Air gets into the cavity and expands the contained space and cannot exit, which prevents expansion of the lung and oxygenation (swells and blocks the airway that spreads oxygen through the lungs).

Pneumothorax can be caused by a blunt chest injury, damage from underlying lung disease, or most horrifying—it may occur for no obvious reason at all. On some occasions, a collapsed lung can be a life-threatening event. The condition leads to hypoxia and cardiopulmonary collapse. The patient can have unilteral decreased or absent breath sounds, acute unilateral chest pain, dysnia, respiratory distress, etc.

A collapse lung will look smaller because of the entrapment of the extra air pushed into the pleura compared to a normal healthy lunch with a thin layer of the pleura. As air pushes against the lung it deviates the trachea to the opposite side, which results in blocking the air passageway.

#### How and who diagnoses pneumothorax?

Pneumothorax is usually diagnosed by a radiologist on a chest x-ray, and can sometimes be very difficult to confirm. An accurate AI algorithm to detect pneumothorax would be useful in a lot of clinical scenarios. AI could be used to triage chest radiographs for priority interpretation, or to provide a more confident diagnosis for non-radiologists.

The Society for Imaging Informatics in Medicine (SIIM) is the leading healthcare organization for those interested in the current and future use of informatics in medical imaging. Their mission is to advance medical imaging informatics across the enterprise through education, research, and innovation in a multi-disciplinary community.

#### How is pneumothorax treated?

Treatment for a pneumothorax usually involves inserting a needle or chest tube between the ribs to remove the excess air (needle decompression). However, a small pneumothorax may heal on its own.

### Challenge

In this competition, you’ll develop a model to classify (and if present, segment) pneumothorax from a set of chest radiographic images. If successful, you could aid in the early recognition of pneumothoraces and save lives. 

### Data and Methods
_____________________________________________________________________________________________

Still in progress... on hold.

### Preprocessing
_____________________________________________________________________________________________

Normalization / 255

### Architectures
_____________________________________________________________________________________________

Testing out U-Net / ResUNet / EfficientNet

### Images
_____________________________________________________________________________________________

![Image](https://github.com/davidtnly/DeepLearning/blob/master/10-pneumothorax-imaging/images-results/sample-image.png)

### Areas to Review
_____________________________________________________________________________________________

+ Working with DICOM files (Extraction, Metadata, Masking, Encoding, Visualizing w/ Label Text)
+ Creating bounding boxes on specified target areas