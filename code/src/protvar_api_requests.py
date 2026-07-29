import requests
import json
import os
import time
import argparse


def input_id(file_path):
    """Retrieve the input ID required for downstream variant queries via the ProtVar API."""
    url = "https://www.ebi.ac.uk/ProtVar/api/input/file?assembly=GRCh38"
    headers = {"accept": "application/json"}
    filename = os.path.basename(file_path)

    with open(file_path, 'rb') as f:
        files = {'file': (filename, f, 'text/plain')}
        print(f"Uploading {filename} to ProtVar...")
        response = requests.post(url, headers=headers, files=files)

    if response.status_code == 200:
        print("Upload Success!")
        return response.json().get('inputId')
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None


def download_request(input_id, email):
    """Request the download of results from the ProtVar API using the input ID."""
    url = "https://www.ebi.ac.uk/ProtVar/api/download"
    headers = {"accept": "application/json"}
    
    payload = {
        'resultId': input_id,
        'pageSize': 1000,
        'assembly': 'GRCh38',
        'function': True,
        'structure': True,
        'email': email,
        'full': True
    }

    print(f"Initiating processing job for input ID: {input_id}...")
    response = requests.post(url, headers=headers, data=payload)

    if response.status_code == 200:
        print(f"Job initiated successfully!")
        return response.json().get('id') 
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None


def check_download_status(request_id):
    """Check the status of the download from the ProtVar API using the request ID."""
    url = "https://www.ebi.ac.uk/ProtVar/api/download/status"
    headers = {"accept": "application/json"}
    
    payload = [request_id]

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        data = response.json()
        return data.get(request_id, {}).get('state') 
    else:
        print(f"Status Check Error {response.status_code}: {response.text}")
        return None


def download_results(request_id, output_filename):
    """Download the final file from the ProtVar API."""
    url = f"https://www.ebi.ac.uk/ProtVar/api/download/{request_id}"
    headers = {"accept": "*/*"}

    print(f"Downloading final dataset to {output_filename}...")
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        with open(output_filename, 'wb') as f:
            f.write(response.content)
        print("Download fully complete and saved!")
        return True
    else:
        print(f"Error {response.status_code}: {response.text}")
        return False



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download ProtVar results via API")
    parser.add_argument("file_path", help="Path to your variant txt file [chr:pos:ref:alt].")
    parser.add_argument("email", help="Your e-mail address to receive the results.")
    parser.add_argument("output_filename", help="Name of the output file to save the results.")
    args = parser.parse_args()

    input_file_id = input_id(args.file_path)
    if not input_file_id:
        exit(1)

    request_id = download_request(input_file_id, args.email)
    if not request_id:
        exit(1)

    filename = os.path.basename(args.file_path)
    print(f"Waiting for ProtVar to process the data for {filename}...")
    while True:
        state = check_download_status(request_id)
        
        if state == "ready":
            print(f"Status: READY. The {filename} has finished processing.")
            break
        else:
            print(f"Status {filename}: {str(state).upper()}. Sleeping for 30 seconds...")
            time.sleep(30)

    output_name = args.output_filename
    if not output_name.endswith(".csv.zip"):
        output_name = output_name + ".csv.zip"
    
    download_results(request_id, output_filename=output_name)