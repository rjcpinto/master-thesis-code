import requests
import sys
  
def request_api(variant_position):
    "Make a request to the Ensembl REST API to retrieve variation data for a specific region."
    variant_chr = variant_position.split(":")[0]
    variant_start = variant_position.split(":")[1]
    variant_end = variant_start
    
    variant_request = f"{variant_chr}:{variant_start}-{variant_end}"

    server = "https://rest.ensembl.org"
    ext = f"/overlap/region/human/{variant_request}?feature=variation"

    r = requests.get(server+ext, headers={ "Content-Type" : "application/json"})
    
    if not r.ok:
        r.raise_for_status() 
    
    decoded = r.json()
    return decoded


def extract_rs_id(decoded_response, variant_position):
    "Extract the rs ID from the API response."
    if len(decoded_response) > 0:
        ref_allele = variant_position.split(":")[2]
        alt_allele = variant_position.split(":")[3]
        
        for variant in decoded_response:
            allele_n = len(variant['alleles'])
            if allele_n != 2:
                if ref_allele == variant['alleles'][0] and alt_allele in variant['alleles'][1:]:
                    return variant['id']
            else:
                if ref_allele == variant['alleles'][0] and alt_allele == variant['alleles'][1]:
                    return variant['id']
                    
        print(f"Variants found, but no allele match for {variant_position}.")
        
    else:
        print(f"No variation data found for {variant_position}.")
    

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 ensembl_api.py <variant_position>")
        print("Example: python3 ensembl_api.py 1:123456:A:T")
        sys.exit(1)

    variant_position = sys.argv[1]
    
    try:
        decoded_response = request_api(variant_position)
        rs_id = extract_rs_id(decoded_response, variant_position)
        print(rs_id, variant_position)
    except Exception as e:
        print(f"API Request failed: {e}")