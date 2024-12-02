from flask import Flask, request, render_template, send_from_directory
import os
from finalMultiScale import analyze  # Import the analyze function from your script

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_FOLDER'] = 'static'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])
if not os.path.exists(app.config['STATIC_FOLDER']):
    os.makedirs(app.config['STATIC_FOLDER'])

@app.route('/', methods=['GET', 'POST'])
def upload_files():
    if request.method == 'POST':
        schools_file = request.files['schools']
        tracts_file = request.files['tracts']
        race_file = request.files['race']

        # Define file paths
        schools_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'Schools_High.csv')
        tracts_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'Census2020_Tracts.csv')
        race_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'RaceByCounty.csv')

        # Save uploaded files to the server
        schools_file.save(schools_filepath)
        tracts_file.save(tracts_filepath)
        race_file.save(race_filepath)

        # Call the analyze function directly and pass the file paths
        analyze(schools_filepath, tracts_filepath, race_filepath)

        return render_template('results.html')

    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
