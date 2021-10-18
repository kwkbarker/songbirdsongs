import csv
import sys
from urllib import request, error
import shutil
import json
import os
import ssl


def import_species_list(species):
    SPECIES_LIST = {}
    fields = ["num","name"]
    with open(species) as csvfile:
        f = csv.DictReader(csvfile, fields)
        for row in f:
            SPECIES_LIST[int(row["num"])] = row["name"].replace('"','')
    return SPECIES_LIST

def main():
    # Error checking - argv input
    # if len(sys.argv) != 2:
    #     return 
    SPECIES_LIST = import_species_list(sys.argv[1])

    for idx in range(1,len(SPECIES_LIST)):
        print(SPECIES_LIST[idx])
        try:
            download(SPECIES_LIST[idx], idx)
        except KeyError:
            print(f"No listings for {SPECIES_LIST[idx]}.")

# adapted from https://github.com/ntivirikin/xeno-canto-py.git by Nazariy Tivirikin

# Disable certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

def metadata(filt):
    page = 1
    page_num = 1
    filt_path = list()
    filt_url = list()
    print("Retrieving metadata...")

    # Scrubbing input for file name and url
    for f in filt:
        filt_url.append(f.replace(' ', '%20'))
        filt_path.append((f.replace(' ', '')).replace(':', '_').replace("\"",""))

    path = 'dataset/metadata/' + ''.join(filt_path)

    # Overwrite metadata query folder 
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

    # Save all pages of the JSON response    
    while page < page_num + 1:
        url = 'https://www.xeno-canto.org/api/2/recordings?query={0}&page={1}'.format('%20'.join(filt_url), page)
        try:
            r = request.urlopen(url)
        except error.HTTPError as e:
            print('An error has occurred: ' + str(e))
            exit()
        print("Downloading metadate page " + str(page) + "...")
        data = json.loads(r.read().decode('UTF-8'))
        filename = path + '/page' + str(page) + '.json'
        with open(filename, 'w') as saved:
            json.dump(data, saved)
        page_num = data['numPages']
        page += 1

    # Return the path to the folder containing downloaded metadata
    return path

# Retrieves metadata and audio recordings
# adapted from https://github.com/ntivirikin/xeno-canto-py.git by Nazariy Tivirikin
def download(filt, species_number):
    page = 1
    page_num = 1
    print("Downloading all recordings for query...")

    # Retrieve metadata to parse for download links
    path = metadata(filt)
    with open(path + '/page' + str(page) + ".json", 'r') as jsonfile:
        data = jsonfile.read()
    data = json.loads(data)
    page_num = data['numPages']
    print("Found " + str(data['numRecordings']) + " recordings for given query, downloading...") 
    while page < page_num + 1:

        # Pulling species name, track ID, and download link for naming and retrieval
        # while i < range(len)


        for i in range(len((data['recordings']))):
            url = 'http:' + data['recordings'][i]['file']
            name = (data['recordings'][i]['en']).replace(' ', '')
            track_id = data['recordings'][i]['id']
            # altered from original code - data saved in numbered directory in the
            # 'data' directory. the number is generated from the existing bird species
            # loaded and mapped into the BIRDS.csv (and loaded into the global variable BIRDS).
            audio_path = 'data/' + species_number + '/'
            audio_file = track_id + '.mp3'
            if not os.path.exists(audio_path):
                os.makedirs(audio_path)

            # If the file exists in the directory, we will skip it
            elif os.path.exists(audio_path + audio_file):
                continue
            print("Downloading " + track_id + ".mp3...")
            request.urlretrieve(url, audio_path + audio_file)
        page += 1

if __name__=="__main__":
    main()
