from functions import greyscaler,sharpener

test_path = "test_cloud.jpg"
# this is an image I took myself!

grey_array = greyscaler(test_path)
final_image = sharpener(grey_array,test_path)

