import cv2
import numpy
from os import listdir
from os.path import isdir, join, isfile
import os
from keras.models import load_model
from flask import Flask, jsonify
from flask import make_response
from flask import request, url_for
from PIL import Image
from werkzeug.utils import secure_filename
import os.path
import wikipedia
from numpy import asarray

app = Flask(__name__)


leaf_names=['Acca sellowiana', 'Acer negundo', 'Acer palmaturu', 'Aesculus californica', 'Alnus sp', 
'Arisarum vulgare', 'Betula pubescens', 'Bougainvillea sp', 'Buxus sempervirens', 'Castanea sativa', 
'Celtis sp', 'Chelidonium majus', 'Corylus avellana', 'Crataegus monogyna', 'Erodium sp', 'Euonymus japonicus', 
'Fragaria vesca', 'Fraxinus sp', 'Geranium sp', 'Hydrangea sp', 'Ilex aquifolium', 'Ilex perado ssp azorica', 
'Magnolia grandiflora', 'Magnolia soulangeana', 'Nerium oleander', 'Papaver sp', 'Pinus sp', 'Podocarpus sp', 
'Polypodium vulgare', 'Populus alba', 'Populus nigra', 'Primula vulgaris', 'Pseudosasa japonica', 'Quercus robur', 
'Quercus suber', 'Salix atrocinerea', 'Schinus terebinthifolius', 'Taxus bacatta', 'Tilia tomentosa',
 'Urtica dioica']


# Loading our model
classifier = load_model("leafs.h5")

# Main route for uploading the image
@app.route('/')
def index():
    return '''
        <form method='POST' action='/upload' enctype='multipart/form-data'>
        <input type="text" name="username">
        <input type="file" name="leaf_image">
        <input type="submit">

    '''

#Post route
@app.route('/upload', methods=['POST'])
def create():
    if 'leaf_image' in request.files:
        leaf_image = request.files['leaf_image']
        
        img = Image.open(leaf_image)
        np_im = asarray(img)
        

        # Altering the image for the model
        input_im = cv2.resize(np_im, (224,224), interpolation=cv2.INTER_LINEAR)
        input_im = input_im /255.
        input_im  = input_im.reshape(1,224,224,3)
        

        # Model predicts the image name
        res = numpy.argmax(classifier.predict(input_im,1,verbose=0),axis=1)
        check_mongo = leaf_names[res[0]]
        print(check_mongo)

        # Scrape the info from wikipedia
        info = wikipedia.summary(check_mongo)
        page = wikipedia.page(check_mongo)
        image_url = page.images[0]
        links = page.links[0]
        return jsonify({"name": check_mongo, "info":info, "image_url": image_url})

            
if __name__ == '__main__':
    app.run(debug=False,threaded=False)









