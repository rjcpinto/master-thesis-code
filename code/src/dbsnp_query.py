import sys
import json
import urllib.request
import urllib.error
import time

# --- CONFIGURATION ---
SERVER = "https://rest.ensembl.org"
# ---------------------

def fetch_ensembl(rs_id):
    clean_id = rs_id.strip()
    if not clean_id.startswith("rs"):
        clean_id = "rs" + clean_id

    ext = f"/variation/human/{clean_id}?content-type=application/json"
    
    try:
        req = urllib.request.Request(SERVER + ext)
        with urllib.request.urlopen(req) as response:
            if response.getcode() == 200:
                data = json.loads(response.read().decode())
                
                current_id = data.get("name", clean_id)
                status = "Merged" if current_id != clean_id else "Active"
                
                # 2. Find Mappings
                mappings = data.get("mappings", [])
                results = []
                
                found_grch38 = False
                
                for m in mappings:
                    # Ensembl uses "GRCh38" or "hg38"
                    if m.get("assembly_name") == "GRCh38":
                        found_grch38 = True
                        
                        chrom = m.get("seq_region_name")
                        pos = m.get("start")
                        allele_string = m.get("allele_string") # e.g. "C/T"
                        
                        # Parse Ref/Alt
                        # Ensembl usually lists REF first in the string for forward strand
                        parts = allele_string.split("/")
                        
                        if len(parts) >= 2:
                            ref = parts[0]
                            alts = parts[1:] # There might be multiple alts
                            
                            for alt in alts:
                                results.append(f"{clean_id},{current_id},{status},{chrom},{pos},{ref},{alt}")
                        else:
                            # Edge case: non-variant or fixed
                            results.append(f"{clean_id},{current_id},{status},{chrom},{pos},{allele_string},NA")
                
                if not found_grch38:
                    return [f"{clean_id},{current_id},NotMappedToGRCh38,NA,NA,NA,NA"]
                
                return results

    except urllib.error.HTTPError as e:
        if e.code == 404: # Not found in Ensembl
            return [f"{clean_id},NA,NotFound,NA,NA,NA,NA"]
        return [f"{clean_id},NA,Error_{e.code},NA,NA,NA,NA"]
    except Exception as e:
        return [f"{clean_id},NA,Error_{str(e)},NA,NA,NA,NA"]

def main(input_file, output_prefix):
    print(f"Fetching from Ensembl API...")
    
    with open(input_file, 'r') as f_in, open(f"{output_prefix}.csv", 'w') as f_out:
        f_out.write("query_id,current_id,status,chr,pos,ref,alt\n")
        
        lines = f_in.readlines()
        total = len(lines)
        
        for i, line in enumerate(lines):
            rs_id = line.strip()
            if not rs_id: continue
            
            rows = fetch_ensembl(rs_id)
            for row in rows:
                f_out.write(row + "\n")
            
            sys.stdout.write(f"\rProgress: {i+1}/{total} - {rs_id}")
            sys.stdout.flush()
            
            time.sleep(0.1)

    print(f"\nDone! Saved to {output_prefix}.csv")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_file> <output_prefix>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_prefix = sys.argv[2]

    main(input_file, output_prefix)