Return the stroke probability map for the continuous CNN model.

# stroke probability map

## input:   
data(type:numpy array)(shape:time * 2)  
model(sklearn model or pytorch model)  
flatten(type: bool)(whether to flatten the input as 200 or use 100*2 as the model input)  
 
## output:   
probanility_map(shape:(number of segments, 12) = ((1+2+4+8+3+6+9), 12)) = (33, 12))
split_list = [1, 2, 4, 8, 3, 6, 9]

# stroke probability map 1   
cut different window lengths with the same number of windows, and predict probability

## input:   
data(type:numpy array)(shape:1024 * 2)   
model(sklearn model or pytorch model)    
flatten(type: bool)(whether to flatten the input as 200 or use 100*2 as the model input)    
model_type(type: int): decide how to predict probability(mostly because of the difference of sklearn and pytorch)   

## output:   
probanility_map(16, 12*3)   
