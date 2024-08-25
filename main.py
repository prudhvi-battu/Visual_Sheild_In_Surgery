import cap
import glove
import mask
import ppe

input_image_path = input('Path of the image to be predicted: ')

c = cap.predict_cap(input_image_path)
g = glove.predict_glove(input_image_path)
m = mask.predict_mask(input_image_path)
p = ppe.predict_ppe(input_image_path)

if c and g and m and  p == 1:
    print("Not Following all the precautions")
else:
    print("Following  all the precautions") 