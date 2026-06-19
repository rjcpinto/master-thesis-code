import requests
import argparse
import sys
import time

def retrieve_functional_anno(line):
    """Retrieve functional annotations for each variant in the DataFrame via ProtVar API."""
    columns = line.strip().split()
    variant = columns[0]
    accession = columns[1]
    position = int(float(columns[2]))  
    variantAA = columns[3].split('/')[1]  

    url = f"https://www.ebi.ac.uk/ProtVar/api/function/{accession}/{position}?variantAA={variantAA}"
    headers = {"accept": "application/json"}
            
    # Send status logs to STDERR so they don't corrupt your output file
    print(f"Retrieving functional annotation for {variant}...", file=sys.stderr)
    
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        # Send errors to STDERR as well
        print(f"Error in variant {variant}: {response.status_code}: {response.text}", file=sys.stderr)
        return None


def parse_functional_anno(response_data):
    """Parse the functional annotation response data to extract relevant information."""
    
    annotation_info = {
        "conservation_score": None,
        "esm1b": None,
        "eve": None,
        "popeve": None,
        "foldxDdg": None,
        "foldxplddt": None
    }
    
    if response_data:
        conserv_data = response_data.get("conservScore")
        if conserv_data:
            annotation_info["conservation_score"] = conserv_data.get("score")
            
        esm_data = response_data.get("esmScore")
        if esm_data:
            annotation_info["esm1b"] = esm_data.get("score")
        
        eve_data = response_data.get("eveScore")
        if eve_data:
            annotation_info["eve"] = eve_data.get("score")
            
        popeve_data = response_data.get("popEveScore")
        if popeve_data:
            annotation_info["popeve"] = popeve_data.get("popeve")
        
        foldxs_data = response_data.get("foldxs")
        if foldxs_data:
            if isinstance(foldxs_data, list) and len(foldxs_data) > 0:
                annotation_info["foldxDdg"] = foldxs_data[0].get("foldxDdg")
                annotation_info["foldxplddt"] = foldxs_data[0].get("plddt")
            elif isinstance(foldxs_data, dict):
                annotation_info["foldxDdg"] = foldxs_data.get("foldxDdg")
                annotation_info["foldxplddt"] = foldxs_data.get("plddt")
    
    annotation_string = ";".join([str(v) if v is not None else "None" for v in annotation_info.values()])
    return annotation_string


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieve functional annotations for variants using ProtVar API.")
    parser.add_argument("line", help="String in the format -> variant format: <chromosome>:<position>:<ref>:<alt>; UniProt accession; AA position; AA change).")
    args = parser.parse_args()
    
    variant_id = args.line.strip().split()[0]
    
    response_data = retrieve_functional_anno(args.line)    
    annotation_string = parse_functional_anno(response_data)
    
    print(f"{variant_id};{annotation_string}")