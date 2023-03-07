#  <span style="color:tan"> **Module 21 Deep Learning Challenge Report**  </span>
### Chris Gruenhagen 6Mar2023
---

**Overview** 

The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. The purpose of this project is to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

**Results:** 

* Data Preprocessing

The following variables are available in the charity_data.csv file

* The variable "IS_SUCCESSFUL" is the target for the model.
* The variables "EIN" and "NAME" are non-beneficial columns and should be removed from the model.  
* All other variables are features to be used to build the model:
    * APPLICATION_TYPE
    * AFFILIATION            
    * CLASSIFICATION           
    * USE_CASE               
    * ORGANIZATION             
    * STATUS                 
    * INCOME_AMT                
    * SPECIAL_CONSIDERATIONS   
    * ASK_AMT 

* Compiling, Training, and Evaluating the Model

I was not able to achieve the target model performance of accuracy >=0.75.
I tried varying number of neurons, layers, activation functions and two different formats for bucketing the initial data.  

| Model Run | N Layers | Activation Functions | N Neurons | Comments |
|-----------|----------|----------------------|-----------|----------|
| Initial (nn) | 2 | relu | 8,5 | starting point from homework |
| 2nd run (nnA) | 3 | relu, tanh, relu | 20,20,10 | increase layers, sandwich activation functions, increase neurons |
| 3rd run (nnaB) | 2 | relu | 20,20 | different bucketing of initial data |
| 4th run (nnaC) | 5 | relu | all 100 | increase layers, increase neurons |
| 5th run (nnaD) | 2 | relu | 100,50 | decrease layers |
---

**Summary:** 

After five attempts at fitting the model, the maximum accuracy achieved was 0.7324 with a loss of 0.5863.  Further evaluation of the initial data may suggest a better binning method for the data that may increase model accuracy. Additional options include trying different optimizers or increasing the training set size. 
