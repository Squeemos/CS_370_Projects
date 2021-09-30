import streamlit as st
import random
import numpy as np
import PPM_Image

# This allows you to select betweeen modifying one image and two
page = st.sidebar.selectbox("Which project would you like to run?",("Interact with one image","Interact with two images"))

# This makes it so when you load an image once it's faster next time
@st.cache(suppress_st_warning=True)
def load_image(path):
    ppm = PPM_Image.PPM_Image()
    ppm.create_from_string(path)
    return ppm

# Interacting with one image
if page == "Interact with one image":
    # Gets the file to be worked on
    path1 = st.sidebar.file_uploader("Image",type="ppm")

    # When the file is loaded
    if path1 is not None:
        # Read it as a string and load a PPM_Image from it
        file_as_string = path1.read().decode("utf-8")
        img = load_image(file_as_string)

        # This just shows the info of the loaded image
        # Doesn't upadte since it's the image in storage (unless you choose a different image)
        st.sidebar.header("Image info")
        st.sidebar.write(f"Image Dimensions: {img.img_size}")
        st.sidebar.write(f"Image max color value: {img.max_color}")


        # These are the operations you can do on the image
        st.sidebar.header("Image operations")
        inv = st.sidebar.checkbox("Image Negative (Inverse)")
        if inv:
            img = img.inverse()

        log = st.sidebar.checkbox("Log Transform")
        if log:
            con = float(st.sidebar.text_input("Please enter the constant",value="1",key="log"))

            img = img.log_transform(con)

        power = st.sidebar.checkbox("Power (Gamma) Transform")
        if power:
            c = float(st.sidebar.text_input("Please enter the constant",value="1",key="gamma2"))
            g = float(st.sidebar.text_input("Please enter the gamma",value="1"))

            img = img.gamma_transform(c,g)

        # Here the operation choices get a little more complicated
        resize = st.sidebar.checkbox("Resize")
        if resize:
            # Allows you to choose method of resizing
            how = st.sidebar.radio("Method of interpolation",("Nearest Neighbor","Bilinear"))

            w = st.sidebar.number_input("Enter a new width",value=img.img_width)
            h = st.sidebar.number_input("Enter a new height",value=img.img_height)

            # Does resizing based on method with new width/height
            if how == "Nearest Neighbor":
                img = img.nearest_neighbor_interpolate(w,h)
            elif how == "Bilinear":
                img = img.bilinear_interpolation(w,h)

        # Enables selection of connected component and how
        connected = st.sidebar.checkbox("Connected component labelling")
        if connected:
            method = st.sidebar.radio("Which method of connected component labelling",("4-connected","8-connected","m-connected"))

            if method == "4-connected":
                img = img.four_connected()
            elif method == "8-connected":
                st.title(f"{method} is not implemented")
            elif method == "m-connected":
                st.title(f"{method} is not implemented")

        # This displays the image to size
        # Note: It does scale with browser size (can't be controlled afaik)
        # However it does keep aspect ratio (show in actual size) if it doesn't outscale the browser
        st.image(img.internal_array,width=None)

        # This allows you to download the image result
        st.sidebar.header("Image download")
        filename = st.sidebar.text_input("Enter filename (exlcuding .ppm) to download as",value=path1.name[:-4])
        st.sidebar.download_button("Download the resulting image",img.write_to_string(),file_name=filename + ".ppm")

# Interact with two images page
if page == "Interact with two images":
    # Option to load both images
    path1 = st.sidebar.file_uploader("First image",type="ppm")
    path2 = st.sidebar.file_uploader("Second image",type="ppm")

    # Load and cache both of the results
    if path1 is not None:
        file_as_string = path1.read().decode("utf-8")
        img1 = load_image(file_as_string)

    if path2 is not None:
        file_as_string = path2.read().decode("utf-8")
        img2 = load_image(file_as_string)

    # When two images are loaded
    if path1 is not None and path2 is not None:
        op = st.sidebar.radio("Image opteration",("None","Addition","Subtraction","Product"))

        # If there's no image operation to be done, just display both of them
        if op == "None":
            st.image(img1.internal_array,width=None)
            st.image(img2.internal_array,width=None)

        # Time to apply an operation
        elif op != "None":
            # If the images are different sizes, we need to resize to common size
            if img1.img_size != img2.img_size:
                # First we choose a method
                method = st.sidebar.radio("Method to resize to common size",("Nearest Neighbor","Bilinear"))

                # Then we look to see which image needs to be scaled up to preserve the most information
                if img1.img_height * img1.img_width >= img2.img_height * img2.img_width:
                    if method == "Nearest Neighbor":
                        img2 = img2.nearest_neighbor_interpolate(img1.img_width,img1.img_height)
                    elif method == "Bilinear":
                        img2 = img2.bilinear_interpolation(img1.img_width,img1.img_height)
                else:
                    if method == "Nearest Neighbor":
                        img1 = img1.nearest_neighbor_interpolate(img2.img_width,img2.img_height)
                    elif method == "Bilinear":
                        img1 = img1.bilinear_interpolation(img2.img_width,img2.img_height)

            # Perform addition
            if op == "Addition":
                met = st.sidebar.radio("Method to process domain",("Clamp","Renormalize"))
                img3 = img1.addition(img2,met.lower())

            # Perform subtraction
            if op == "Subtraction":
                met = st.sidebar.radio("Method to process domain",("Clamp","Renormalize"))
                img3 = img1.subtraction(img2,met.lower())

            # Perform product
            if op == "Product":
                img3 = img1.product(img2)

            # Display the image
            st.image(img3.internal_array,width=None)

            # Ability to download the image
            st.sidebar.header("Image download")
            filename = st.sidebar.text_input("Enter filename (exlcuding .ppm) to download as",value="Result")
            st.sidebar.download_button("Download the resulting image",img3.write_to_string(),file_name=filename + ".ppm")
