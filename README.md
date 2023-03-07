#  <span style="color:tan"> **Module 21 Deep Learning Challenge**  </span>
### Chris Gruenhagen 28Feb2023
---

## **Purpose**

    Use Deep Learning to create a binary classifier that can predict whether applicants will be successful if they are funded by Alphabet Soup.

## **Background**

    The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.  

## **Results** 

* Data Preprocessing

The following variables were available in the starting data from the charity_data.csv file. Data processing available in AlphabetSoupCharity_Optimization.ipynb.

* The variable "IS_SUCCESSFUL" is the target for the model.
* The variables "EIN" and "NAME" are non-beneficial columns and were removed from the model.  
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

| Model Run | N Layers | Activation Functions | N Neurons | Comments | Accuracy | Loss |
|-----------|----------|----------------------|-----------|----------|-----------|--------|
| Initial (nn) | 2 | relu | 8,5 | starting point from homework | 0.7284 | 0.5523 |
| 2nd run (nnA) | 3 | relu, tanh, relu | 20,20,10 | increase layers, sandwich activation functions, increase neurons | 0.7298 | 0.5562 |
| 3rd run (nnaB) | 2 | relu | 20,20 | different bucketing of initial data | 0.7259 | 0.5500 |
| 4th run (nnaC) | 5 | relu | all 100 | increase layers, increase neurons | 0.7324 | 0.5863 |
| 5th run (nnaD) | 2 | relu | 100,50 | decrease layers | 0.7306 | 0.5635 |
---

## **Summary** 

After five attempts at fitting the model, the maximum accuracy achieved was 0.7324 with a loss of 0.5863.  Further evaluation of the initial data may suggest a better binning method for the data that may increase model accuracy. Additional options include trying different optimizers or increasing the training set size. 

---

# **Continue the Data Preparation Cheat Sheet**
Data Preparation Cheat Sheet
1. check datatypes (numeric cols may have non-numeric info that needs to be removed, data types may need to be changed) 
    df.dtypes

2. replace categorical with numeric values - consider replacing empty strings with NaN 
    https://www.geeksforgeeks.org/how-to-convert-categorical-variable-to-numeric-in-pandas/


    df['colname'].unique()
    df['colname'].replace(['uniquevalue1', 'uniquevalue2','uniquevalue3',''],
                        [1,2,3,0], inplace=True)

    OR
    
    dummies = pd.get_dummies(df.colname)
    merged = pd.concat([df,dummies].axis='columns')
    merged.drop('colname',axis='columns')

    OR (example from homework Mod20-Day1-Activity02)

    df['colname'].unique()
    colname_dict = {'uniquevalue1': 1, 'uniquevalue2': 2, 'uniquevalue3': 3}
    df2 = df.replace({'colname': colname_dict})
     note: not sure how this would handle a null

    OR (example from homework Mod20-Day1-Activity03)

    def changeChannel(channel):
        if channel == "uniquevalue1":
            return 1
        if channel == "uniquevalue2":
            return 2
        if channel == "uniquevalue3":
            return 2        
        else:
            return 0 
    df_shopping["Channel"] = df_shopping["Channel"].apply(changeChannel)

3. check for nas, nulls and remove if necessary  (na = NOT A NUMBER)  NOTE: df.isna() == df.isnull()  
    https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dropna.html

    df.isna().sum()
    df.dropna() 

    NOTE: This does not detect empty strings.  To detect empty strings search for df['colname']==''
    https://stackoverflow.com/questions/27159189/find-empty-or-nan-entry-in-pandas-dataframe

4. check for duplicate values and remove if necessary
    https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop_duplicates.html

    df.duplicated().sum()
    df.drop_duplicates(keep='first')  note: keep = 'first' is default


5. drop unnecessary columns 
    https://www.w3schools.com/python/pandas/ref_df_drop.asp#:~:text=The%20drop()%20method%20removes

    df.drop(columns=['colname'])
    or
    df.drop('colname',axis = 1)
    df.drop('colname',axis = 'columns')

6. standardize dataset so larger numbers don't influence the outcome more
    Scale the data - all columns (example from homework Mod20-Day1-Activity06)
        df_scaled = StandardScaler().fit_transform(df)

    Scale the data - subset of columns (example from homework Mod20-Day1-Activity03)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_shopping[['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicatessen']])

        If you need to add back a column that wasn't part of the scaling (was already 0,1)

        # A list of the columns from the original DataFrame
        df_shopping.columns

        # Create a DataFrame with the transformed data
        new_df_shopping = pd.DataFrame(scaled_data, columns=df_shopping.columns[1:])
        new_df_shopping['Channel'] = df_shopping['Channel']
        new_df_shopping.head()

---

#  &#x1f52e; &#x1f520; &#x1f523; 🔢 📈 &#x1F469;&#x200d;&#x1F4bb; 📉 🔢 &#x1f523; &#x1f520; &#x1f52e;
     emoji & format references:
        https://emojipedia.org
        https://commons.wikimedia.org/wiki/Emoji/Table
        https://www.markdownguide.org/basic-syntax/

        
***See below for original homework instructions***
# <span style="color:tan"> Module 21 Deep Learning Challenge</span>
Due March 6, 2023 by 11:59pm 

Points 100 

Submitting a text entry box or a website url
_________________________________________________________
## <span style="color:red"> **Background**  </span>

The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, you have received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:

* EIN and NAME—Identification columns
* APPLICATION_TYPE—Alphabet Soup application type
* AFFILIATION—Affiliated sector of industry
* CLASSIFICATION—Government organization classification
* USE_CASE—Use case for funding
* ORGANIZATION—Organization type
* STATUS—Active status
* INCOME_AMT—Income classification
* SPECIAL_CONSIDERATIONS—Special considerations for application
* ASK_AMT—Funding amount requested
* IS_SUCCESSFUL—Was the money used effectively

##  <span style="color:orange"> **Before You Begin** </span>
1. Create a new repository for this project called deep-learning-challenge. Do not add this Challenge to an existing repository.
2. Clone the new repository to your computer.
3. Inside your local git repository, create a directory for the Deep Learning Challenge.
4. Push the above changes to GitHub.

##  <span style="color:yellow"> **Files** </span>

Download the following files to help you get started:

<a href = "https://courses.bootcampspot.com/courses/2584/assignments/40504?module_item_id=771767 " target = "_blank"> Module 21 Challenge files </a>


##  <span style="color:green">  **Instructions** </span>

### ***Step 1: Preprocess the Data***
Using your knowledge of Pandas and scikit-learn’s StandardScaler(), you’ll need to preprocess the dataset. This step prepares you for Step 2, where you'll compile, train, and evaluate the neural network model.

Using the information we provided in the Challenge files, follow the instructions to complete the preprocessing steps.

1. Read in the charity_data.csv to a Pandas DataFrame, and be sure to identify the following in your dataset:
    *   What variable(s) are the target(s) for your model?
    *   What variable(s) are the feature(s) for your model?
2. Drop the EIN and NAME columns.

3. Determine the number of unique values for each column.

4. For columns that have more than 10 unique values, determine the number of data points for each unique value.

5. Use the number of data points for each unique value to pick a cutoff point to bin "rare" categorical variables together in a new value, Other, and then check if the binning was successful.

6. Use pd.get_dummies() to encode categorical variables.

7. Split the preprocessed data into a features array, X, and a target array, y. Use these arrays and the train_test_split function to split the data into training and testing datasets.

8. Scale the training and testing features datasets by creating a StandardScaler instance, fitting it to the training data, then using the transform function.

### ***Step 2: Compile, Train, and Evaluate the Model***
Using your knowledge of TensorFlow, you’ll design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup-funded organization will be successful based on the features in the dataset. You’ll need to think about how many inputs there are before determining the number of neurons and layers in your model. Once you’ve completed that step, you’ll compile, train, and evaluate your binary classification model to calculate the model’s loss and accuracy.

1. Continue using the Jupyter Notebook in which you performed the preprocessing steps from Step 1.

2. Create a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.

3. Create the first hidden layer and choose an appropriate activation function.

4. If necessary, add a second hidden layer with an appropriate activation function.

5. Create an output layer with an appropriate activation function.

6. Check the structure of the model.

7. Compile and train the model.

8. Create a callback that saves the model's weights every five epochs.

9. Evaluate the model using the test data to determine the loss and accuracy.

10. Save and export your results to an HDF5 file. Name the file AlphabetSoupCharity.h5.

### ***Step 3: Optimize the Model***
Using your knowledge of TensorFlow, optimize your model to achieve a target predictive accuracy higher than 75%.

Use any or all of the following methods to optimize your model:

* Adjust the input data to ensure that no variables or outliers are causing confusion in the model, such as:
    * Dropping more or fewer columns.
    * Creating more bins for rare occurrences in columns.
    * Increasing or decreasing the number of values for each bin.
* Add more neurons to a hidden layer.
* Add more hidden layers.
* Use different activation functions for the hidden layers.
* Add or reduce the number of epochs to the training regimen.

**Note:** If you make at least three attempts at optimizing your model, you will not lose points if your model does not achieve target performance.

1. Create a new Jupyter Notebook file and name it AlphabetSoupCharity_Optimization.ipynb.

2. Import your dependencies and read in the charity_data.csv to a Pandas DataFrame.

3. Preprocess the dataset as you did in Step 1. Be sure to adjust for any modifications that came out of optimizing the model.

4. Design a neural network model, and be sure to adjust for modifications that will optimize the model to achieve higher than 75% accuracy.

5. Save and export your results to an HDF5 file. Name the file AlphabetSoupCharity_Optimization.h5.

### ***Step 4: Write a Report on the Neural Network Model***
For this part of the assignment, you’ll write a report on the performance of the deep learning model you created for Alphabet Soup.

The report should contain the following:

1. **Overview** of the analysis: Explain the purpose of this analysis.

2. **Results:** Using bulleted lists and images to support your answers, address the following questions:

* Data Preprocessing

    * What variable(s) are the target(s) for your model?
    * What variable(s) are the features for your model?
    * What variable(s) should be removed from the input data because they are neither targets nor features?
* Compiling, Training, and Evaluating the Model

    * How many neurons, layers, and activation functions did you select for your neural network model, and why?
    * Were you able to achieve the target model performance?
    * What steps did you take in your attempts to increase model performance?
3. **Summary:** Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and then explain your recommendation.

##  <span style="color:blue"> **Requirements** </span>
### ***Preprocess the Data (30 points)***
* Create a dataframe containing the charity_data.csv data , and identify the target and feature variables in the dataset (2 points)
* Drop the EIN and NAME columns (2 points)
* Determine the number of unique values in each column (2 points)
* For columns with more than 10 unique values, determine the number of data points for each unique value (4 points)
* Create a new value called Other that contains rare categorical variables (5 points)
* Create a feature array, X, and a target array, y by using the preprocessed data (5 points)
* Split the preprocessed data into training and testing datasets (5 points)
* Scale the data by using a StandardScaler that has been fitted to the training data (5 points)

### ***Compile, Train and Evaluate the Model (20 points)***
* Create a neural network model with a defined number of input features and nodes for each layer (4 points)
* Create hidden layers and an output layer with appropriate activation functions (4 points)
* Check the structure of the model (2 points)
* Compile and train the model (4 points)
* Evaluate the model using the test data to determine the loss and accuracy (4 points)
* Export your results to an HDF5 file named AlphabetSoupCharity.h5 (2 points)

### ***Optimize the Model (20 points)***
* Repeat the preprocessing steps in a new Jupyter notebook (4 points)
* Create a new neural network model, implementing at least 3 model optimization methods (15 points)
* Save and export your results to an HDF5 file named AlphabetSoupCharity_Optimization.h5 (1 point)

## <span style="color:indigo"> **References**  </span>
IRS. Tax Exempt Organization Search Bulk Data Downloads. <a href = 'https://www.irs.gov/charities-non-profits/tax-exempt-organization-search-bulk-data-downloads' target = '_blank'>  https://www.irs.gov/  </a>

© 2023 edX Boot Camps LLC
