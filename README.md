# Non_file_virus_detected
### We use the around 200*[10 to 20] feature to solve this problem and finally achieve to 94% in private. (Top 11% in Competition)
### Most important feature as follow:
### 1. the mean ,std ,diff of time for each file
### 2. discard the outlier of USERID
### 3. separate the time interval into different scale, like week,date,hour
### 4. group by file and calculate each Product ID
### 5. https://hackmd.io/C0YofbLXRce5S6jfQAX1sg
### 6. we predict the model from two ways: XGB Classifier and LSTM, recently LSTM can achieve 91% around.


