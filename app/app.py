from flask import Flask, request, render_template, send_from_directory
import os
import subprocess

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

        schools_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'Schools_High.csv')
        tracts_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'Census2020_Tracts.csv')
        race_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'RaceByCounty.csv')

        schools_file.save(schools_filepath)
        tracts_file.save(tracts_filepath)
        race_file.save(race_filepath)

        # Running the existing Python script
        subprocess.run(['python', 'your_script.py', schools_filepath, tracts_filepath, race_filepath], check=True)

        return render_template('upload.html', download=True)
    return render_template('upload.html', download=False)

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    return send_from_directory(app.config['STATIC_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
