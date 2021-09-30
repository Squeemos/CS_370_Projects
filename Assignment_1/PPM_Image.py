import numpy as np
from sklearn.preprocessing import MinMaxScaler

# This function just tries to convert whatever is in the array to an int
# If it can't it skips it
def convert_to_ints(x):
    new_list = []
    for value in x:
        try:
            new_list.append(int(value))
        except:
            continue
    return new_list

class PPM_Image(object):
    # Create PPM_Image from values
    def __init__(self,width=0,height=0,max_color=0,internal=None):
        self.img_width = width
        self.img_height = height
        self.img_size = width,height
        self.max_color = max_color
        self.internal_array = internal

    def create_from_file(self,filename):
        # Open the file to see the method which it should be read
        with open(filename) as f:
            first_line = f.readline().lower()

            # If it's a P3 image, easy to read
            if 'p3' in first_line:
                print("Reading P3 PPM image")
                try:
                    # Read in the whole file, split into values
                    all_file = f.read()
                    a_f = all_file.replace('\n',' ').split(' ')
                    # Convert everything to an integer
                    a_f_int = convert_to_ints(a_f)

                    # From the list, exctract the dimensions
                    self.img_width = a_f_int[0]
                    self.img_height = a_f_int[1]
                    self.img_size = self.img_width, self.img_height
                    self.max_color = a_f_int[2]

                    print(f"Image dimensions: {self.img_size}")
                    print(f"Max color value: {self.max_color}")

                    # Everything left in the array is pixel values
                    # Convert to proper image format
                    all_pixels = a_f_int[3:]
                    img = np.array(all_pixels).reshape(self.img_height,self.img_width,3)
                    self.internal_array = img.copy()
                    self.internal_array = self.internal_array.astype('uint8')
                except:
                    print(f"Error reading {filename}")

    def create_from_string(self,string):
        # Very similar to creatr from file, but for streamlit we need to use a string instead
        try:
            # Replace all of the newlines with nothing and split on spaces
            a_f = string.replace('\n',' ').split(' ')
            # Convert everything to an array of ints
            a_f_int = convert_to_ints(a_f)
            # Read in specific fields from the list of ints
            self.img_width = a_f_int[0]
            self.img_height = a_f_int[1]
            self.img_size = self.img_width, self.img_height
            self.max_color = a_f_int[2]

            print(f"Image dimensions: {self.img_size}")
            print(f"Max color value: {self.max_color}")

            # Everything left in the array is pixel values
            # Convert to proper image format
            all_pixels = a_f_int[3:]
            img = np.array(all_pixels).reshape(self.img_height,self.img_width,3)
            self.internal_array = img.copy()
            self.internal_array = self.internal_array.astype('uint8')
        except:
            print(f"Error reading from string")

    def inverse(self):
        # Numpy handles value - array, so create a new PPM_Image with max_color - image
        return PPM_Image(self.img_width,self.img_height,self.max_color,self.max_color - self.internal_array)

    def log_transform(self,c):
        # Divide by max_color to convert to L = [0,1]
        new_array = self.internal_array / self.max_color

        # Apply the log transformation
        new_array = c * (np.log10(1 + new_array))

        # Multiply by max value to undo the transformation
        new_array = new_array * self.max_color
        new_array = new_array.astype("uint8")

        return PPM_Image(self.img_width,self.img_height,self.max_color,new_array.copy())

    def gamma_transform(self,c,g):
        # Transform so that L = [0,1]
        new_array = self.internal_array / self.max_color

        # Apply the gamma transformation
        new_array = c * np.power(new_array,g)

        # Undo the trasnformation to L
        new_array = new_array * self.max_color
        new_array = new_array.astype("uint8")

        return PPM_Image(self.img_width,self.img_height,self.max_color,new_array.copy())

    def write_to_string(self):
        # This is so that we can save the image as a file in streamlit
        # Each section creates a string according to PPM format
        first = "P3 \n"
        second = f"{self.img_width} {self.img_height} \n"
        third = f"{self.max_color} \n"

        # This takes the internal array, flattens it down to a 1D array, and converts it to a list
        # Then we convert the list to a string and remove the brackets and commas
        temp = self.internal_array.flatten().tolist()
        fourth = str(temp).replace('[','').replace(']','').replace(',','')

        # This is the entire PPM_Image as a single string which streamlit can save
        return first + second + third + fourth

    def nearest_neighbor_interpolate(self,new_width,new_height):
        # This is thew new array for the image
        new_array = np.zeros((new_height,new_width,3))

        # Loop over each pixel in output space
        for rp in range(new_height):
            for cp in range(new_width):
                # Calculate the nearest neighbor in input space
                row_pos = round((self.internal_array.shape[0] - 1) * (rp / new_height))
                col_pos = round((self.internal_array.shape[1] - 1)  * (cp / new_width))

                # Set the value of the pixel in output space
                new_array[rp,cp] = self.internal_array[row_pos,col_pos]
        new_array = new_array.astype("uint8")

        return PPM_Image(new_width,new_height,self.max_color,new_array.copy())

    def addition(self,other,how):
        # Make sure the images are the same size
        assert self.img_size == other.img_size, f"{self.img_size} and {other.img_size}"
        # Numpy handles array addition (element wise)
        # Divide each array by their max color so that L = [0,1]
        new_array = (self.internal_array / self.max_color) + (other.internal_array / other.max_color)

        # If we clamp the values, just force L = [0,1]
        if how == "clamp":
            new_array[new_array > 1] = 1
            new_array[new_array < 0] = 0

        # If we renormalize, use minmax normalization
        elif how == "renormalize":
            scaler = MinMaxScaler()
            # Requires data to be a single column
            flat = new_array.reshape(-1,1)
            # Fir the scalre and transform the data
            scaler.fit(flat)
            flat = scaler.transform(flat)
            # Reshape the array again
            new_array = flat.reshape(self.internal_array.shape)

        # Convert the array back to standard format
        new_array = self.max_color * new_array
        new_array = new_array.astype("uint8")

        return PPM_Image(self.img_width,self.img_height,self.max_color,new_array.copy())

    def subtraction(self,other,how):
        # Make sure the arrays are the same size
        assert self.img_size == other.img_size
        # Convert the values for each array to be L = [0,1]
        new_array = (self.internal_array / self.max_color) - (other.internal_array / other.max_color)

        # If we clamp force L = [0,1]
        if how == "clamp":
            new_array[new_array > 1] = 1
            new_array[new_array < 0] = 0

        # If we renomalize, use minmax renormalization
        elif how == "renormalize":
            scaler = MinMaxScaler()
            # Requires data to be a single column
            flat = new_array.reshape(-1,1)
            # Fit the scaler and transform the data
            scaler.fit(flat)
            flat = scaler.transform(flat)
            # Reshape the array back into image
            new_array = flat.reshape(self.internal_array.shape)

        # Convert back to standard format
        new_array = self.max_color * new_array
        new_array = new_array.astype("uint8")

        return PPM_Image(self.img_width,self.img_height,self.max_color,new_array.copy())

    def product(self,other):
        # Make sure the images are the same size
        assert self.img_size == other.img_size
        # Numpy handles element wise product
        new_array = self.internal_array * other.internal_array

        return PPM_Image(self.img_width,self.img_height,self.max_color,new_array.copy())

    def bilinear_interpolation(self,new_width,new_height):
        # This is the new image
        new_array = np.zeros((new_height,new_width,3))

        # Loop over each pixel in output space
        for rp in range(new_height):
            for cp in range(new_width):
                # Calculate the value in image space
                x = rp / new_height
                y = cp / new_width

                # Find the value of the previous pixel in input space
                x_prev = int(np.floor((self.internal_array.shape[0] - 1) * x))
                y_prev = int(np.floor((self.internal_array.shape[1] - 1)  * y))

                # Find the value of the next pixel in input space
                x_next = min(x_prev + 1, self.img_height - 1)
                y_next = min(y_prev + 1, self.img_width - 1)

                # Calculate the position of the previous and next pixels in image space
                x1 = x_prev / self.img_height
                x2 = x_next / self.img_height
                y1 = y_prev / self.img_width
                y2 = y_next / self.img_width

                # This is based off the formula from the slides
                a = (x - x1) / (x2 - x1)
                b = (y - y1) / (y2 - y1)

                f_x_y1 = (1 - a) * self.internal_array[x_prev,y_prev] + a * self.internal_array[x_next,y_prev]
                f_x_y2 = (1 - a) * self.internal_array[x_prev,y_next] + a * self.internal_array[x_next,y_prev]
                f_x_y = (1 - b) * f_x_y1 + b * f_x_y2
                new_array[rp,cp] = f_x_y

                new_array = new_array.astype("uint8")

        return PPM_Image(new_width,new_height,self.max_color,new_array.copy())

    def four_connected(self):
        # This is the array that will store the labels of the pixels
        labels = np.full((self.img_height,self.img_width), 0)
        # This is the counter for what "object" we're in
        current_label = 1

        # This is the pairs of labels to be grouped together
        # Each key in the pair will have a set as the value
        # E.G. pairs[1] = {2,3,4,5,...}
        # 1 : {2,3,4,5,...}
        pairs = {}
        # set labels for components
        # Loop over each pixel in the image
        for cp in range(self.img_height):
            for rp in range(self.img_width):
                # if not a background pixel
                if np.all(self.internal_array[cp,rp] == np.array([255,255,255])):
                    # If the pixel above and to the left is within the image
                    if cp - 1 < 0:
                        up = 0
                    else:
                        up = labels[cp-1,rp]
                    if rp - 1 < 0:
                        left = 0
                    else:
                        left = labels[cp,rp-1]

                    # If both up and left are labelled
                    if up != 0 and left != 0:
                        # If up and left are not the same, we need to add them as a pair
                        if up != left:
                            # The value is what will be added on the right
                            value = int(left)
                            # The key is what will be added on the left
                            key = int(up)
                            # If the key is not already in the pairs
                            if key not in pairs:
                                # We say we haven't added the key yet
                                added = False
                                # Loop over every current key in pairs
                                # We're going to be checking the right side of pairs
                                # We check to see if it's on the right, and if it is
                                # Add the value to the set the key is in
                                for k in pairs.keys():
                                    if key in pairs[k]:
                                        pairs[k].add(value)
                                        added = True
                                # If the key doesn't exist anywhere in the pairs
                                # We need to add it since it's a new object
                                if not added:
                                    pairs[key] = {value}
                            # If the key is already in pairs
                            else:
                                # If the value is not on the right, add it
                                if value not in pairs[key]:
                                    pairs[key].add(value)
                        # Set the label of the current pixel to the left value
                        labels[cp,rp] = left

                    # One of the neighbors is 0
                    # Left is zero so we take the label from above
                    elif up != 0 and left == 0:
                        labels[cp,rp] = up
                    # Above is zero so we take the label from left
                    elif up == 0 and left != 0:
                        labels[cp,rp] = left
                    # All neighbors are 0, therefore new object
                    else:
                        labels[cp,rp] = current_label
                        current_label += 1

        # From here we loop over each pair of keys in pairs
        # We're going to check if the sets overlap, and if they do, combine them
        for key1 in pairs.keys():
            for key2 in pairs.keys():
                # Only check when the sets aren't the same
                if key1 != key2:
                    set1 = pairs[key1]
                    set2 = pairs[key2]

                    # If the sets aren't disjoint, combine them
                    # Also add in the current key to the first set to make the "chain"
                    if not set1.isdisjoint(set2):
                        pairs[key1] = set1.union(set2)
                        pairs[key1].add(int(key2))

        # Starting with the largest key, we're going to relabel
        # We do this because we don't necessarily have keys/values (there are values that are also keys)
        # E.G. 1 : 2,3,4,5,6,7....20, 21, 22,23,25 (notice we miss 24)
        # 20: 21,22,23,24 (we now have 24 -> 20)
        # Starting from the back makes sure we set 20->1 so that 24 -> 20 -> 1
        back = sorted(pairs.keys(),reverse=True)
        # Swap all of the labels in the labels array
        for key in back:
            # Get the set that will relabel to the head value
            set_from = pairs[key]
            # For each value in the set, label the value the head value
            for val_from in set_from:
                labels[labels == val_from] = int(key)

        # recolor pixels to label components
        colors = [list(np.random.choice(range(256), size=3)) for _ in range(current_label)]
        colors[0] = np.array([0,0,0])
        # This is the final output image with labelled componenets
        new_array = np.full(self.internal_array.shape,0)
        # Iterate over the image and set the values
        for cp in range(self.img_height):
            for rp in range(self.img_width):
                new_array[cp,rp] = colors[labels[cp,rp]]

        return PPM_Image(self.img_width,self.img_height,self.max_color,new_array.copy())
