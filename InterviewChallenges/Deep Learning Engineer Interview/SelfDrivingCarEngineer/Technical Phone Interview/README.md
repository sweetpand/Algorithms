## Technical Phone Interview
----

1. What is overfitting?   
Ans. 
   - Model remembering the training data and not performing well on the test data is called overfitting.

2. When do you know overfitting has occurred without looking at the test data?  
Ans. 
   - We can use validation set to see if the model overfits the training data.
   - If validation error graph is more then training error graph then model overfits the training data.

   
3. How to prevent over-fitting?  
Ans.
   - We can prevent over-fitting by regularization.

   
4. What are the types of regularization?  
Ans.
   - Dropouts and L2 regularization are some types of regularizations
   
5. What are Dropouts?  
Ans.
   - The signal from some of the nodes are skipped.
   - This prevents weights from exploding or having higher value.
   
6. What is batch-normalization?  
Ans.
	- batch-normalization is a normalization of the batch data being processed
   - If the data in it's raw format is normalized to get good results then same is the idea to normalize the data in the intermediary layers 
	
7. Why do we use batch-normalization?  
Ans.
    - It helps in reducing the training time and getting fast towards the global minima.
