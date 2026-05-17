import sys 
import os 

sys.path.append(
    os.path.abspath('../app')
)

from predict import predict_output

input_image= '../data/chest_xray/test/PNEUMONIA/person1_virus_6.jpeg'

result=predict_output(input_image)
print (result)